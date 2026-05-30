"""Unit tests for profile-cold's modality-aware input factory.

Pins that the factory picks the right primary input for vision/text/
audio/multimodal models so profile-cold is generic across HF model
classes — not just vision.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from scripts.tt_hw_planner.commands.profile_cold import (  # noqa: E402
    _build_input_factory,
    _detect_modality,
)


# ---------------------------------------------------------------------------
# Fixtures: minimal models that look HF-like enough for the factory
# ---------------------------------------------------------------------------


def _make_model(forward_first_param: str, **config_attrs) -> torch.nn.Module:
    """Build a torch.nn.Module whose forward has a specific first-param
    name + a config namespace with the given attributes."""

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(**config_attrs)

        # We dynamically rename the first param via exec to avoid
        # eval/lambda quirks with reserved arg names.

    if forward_first_param == "pixel_values":

        def forward(self, pixel_values):  # type: ignore[no-redef]
            return pixel_values

    elif forward_first_param == "input_ids":

        def forward(self, input_ids):  # type: ignore[no-redef]
            return input_ids

    elif forward_first_param == "input_features":

        def forward(self, input_features):  # type: ignore[no-redef]
            return input_features

    elif forward_first_param == "x":

        def forward(self, x):  # type: ignore[no-redef]
            return x

    else:
        raise ValueError(f"unknown forward_first_param: {forward_first_param}")

    _M.forward = forward  # type: ignore[attr-defined]
    return _M()


# ---------------------------------------------------------------------------
# _detect_modality
# ---------------------------------------------------------------------------


def test_detects_vision_from_pixel_values_signature():
    m = _make_model("pixel_values", image_size=224)
    assert _detect_modality(m) == "vision"


def test_detects_text_from_input_ids_signature():
    m = _make_model("input_ids", vocab_size=50000)
    assert _detect_modality(m) == "text"


def test_detects_audio_from_input_features_signature():
    m = _make_model("input_features", num_mel_bins=80)
    assert _detect_modality(m) == "audio"


def test_falls_back_to_vision_when_first_param_unknown():
    """Unknown first-param name defaults to vision (safe-default)."""
    m = _make_model("x")
    assert _detect_modality(m) == "vision"


def test_multimodal_detected_via_vision_config():
    """A model with config.vision_config (CLIP-style) is multimodal."""

    class _MultiM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                vision_config=SimpleNamespace(image_size=224),
                text_config=SimpleNamespace(vocab_size=32000),
            )

        def forward(self, x):  # ambiguous first-param
            return x

    assert _detect_modality(_MultiM()) == "multimodal"


# ---------------------------------------------------------------------------
# _build_input_factory: shape correctness per modality
# ---------------------------------------------------------------------------


def test_vision_factory_produces_image_tensor():
    m = _make_model("pixel_values", image_size=256)
    factory, label = _build_input_factory(m)
    t = factory(0)
    assert t.shape == (1, 3, 256, 256)
    assert t.dtype == torch.float32
    assert "vision" in label


def test_text_factory_produces_input_ids():
    m = _make_model("input_ids", vocab_size=32000)
    factory, label = _build_input_factory(m)
    t = factory(0)
    assert t.dim() == 2
    assert t.shape[0] == 1  # batch
    assert t.dtype == torch.long
    # Token ids must be valid for the vocab
    assert int(t.max()) < 32000
    assert "text" in label


def test_audio_factory_produces_audio_features():
    m = _make_model("input_features", num_mel_bins=80, max_source_positions=750)
    factory, label = _build_input_factory(m)
    t = factory(0)
    assert t.dim() == 3
    assert t.shape == (1, 80, 1500)  # min(750*2, 3000) = 1500
    assert t.dtype == torch.float32
    assert "audio" in label


def test_factory_seed_varies_inputs():
    """Multi-pass probing relies on varied inputs across passes — same
    seed = same input, different seeds = different inputs."""
    m = _make_model("pixel_values", image_size=224)
    factory, _ = _build_input_factory(m)
    a = factory(0)
    b = factory(1)
    a2 = factory(0)
    # Different seeds produce different tensors
    assert not torch.equal(a, b)
    # Same seed reproduces exact same tensor (deterministic)
    assert torch.equal(a, a2)


def test_text_factory_caps_sequence_length():
    """Text seq_len is capped at 64 even if the model supports more —
    keeps the probe fast."""
    m = _make_model("input_ids", vocab_size=32000, max_position_embeddings=4096)
    factory, label = _build_input_factory(m)
    t = factory(0)
    assert t.shape[1] <= 64
    assert "seq_len=64" in label


def test_vision_image_size_falls_back_when_config_missing():
    """No image_size in config → default 1024."""
    m = _make_model("pixel_values")  # no image_size in config
    factory, _ = _build_input_factory(m)
    t = factory(0)
    assert t.shape == (1, 3, 1024, 1024)
