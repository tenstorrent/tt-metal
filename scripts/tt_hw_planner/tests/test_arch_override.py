"""config.architectures overrides a diffusion/unknown pipeline_tag for causal/MoE transformers.

A model tagged text-to-image but whose config declares *ForCausalLM/*ForCausalMM
(e.g. HunyuanImage-3.0 -> HunyuanImage3ForCausalMM, MoE) must be reclassified to
LLM so architecture/MoE detection runs, instead of early-returning as Image and
force-fitting a diffusion family. Match is on the architecture suffix, not a model
name (model-agnostic), and a genuine diffusion model (no such arch) is untouched.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _f():
    return importlib.import_module("scripts.tt_hw_planner.probe")._arch_override_category


def test_causal_mm_image_reclassified_to_llm():
    f = _f()
    assert f("Image", {"architectures": ["HunyuanImage3ForCausalMM"]}) == "LLM"


def test_causal_lm_variants_and_video_unknown():
    f = _f()
    assert f("Image", {"architectures": ["LlamaForCausalLM"]}) == "LLM"
    assert f("Video", {"architectures": ["FooForCausalMM"]}) == "LLM"
    assert f("Unknown", {"architectures": ["BarForCausalLM"]}) == "LLM"


def test_real_diffusion_untouched():
    f = _f()
    assert f("Image", {"architectures": ["UNet2DConditionModel"]}) == "Image"
    assert f("Image", {"architectures": []}) == "Image"
    assert f("Image", {}) == "Image"


def test_non_candidate_categories_never_touched():
    f = _f()
    # already-transformer categories are left as-is even with a causal arch
    assert f("LLM", {"architectures": ["XForCausalLM"]}) == "LLM"
    assert f("VLM", {"architectures": ["XForCausalLM"]}) == "VLM"
    assert f("STT", {"architectures": ["XForCausalMM"]}) == "STT"


def test_substring_false_positive_guarded():
    f = _f()
    # 'ForCausalMMX' should not match ForCausalMM\b (word boundary)
    assert f("Image", {"architectures": ["WeirdForCausalMMX"]}) == "Image"
