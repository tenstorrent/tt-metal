from __future__ import annotations

import pytest

from scripts.tt_hw_planner.reuse_registry import (
    lookup,
    lookup_by_concept,
    entries_for_model_type,
    all_entries,
    _derived_entries,
)
from scripts.tt_hw_planner.bringup_plan import (
    REUSE,
    ADAPT,
    NEW,
    _extract_components_from_module_tree,
)


class TestRegistryLookup:
    def test_qwen3_rmsnorm_maps_to_common_rmsnorm(self):
        hit = lookup("qwen3", "Qwen3RMSNorm")
        assert hit is not None
        assert hit.status == ADAPT
        assert hit.tt_path == "models/common/rmsnorm.py"
        assert hit.tt_class == "RMSNorm"

    def test_qwen3_attention_maps_to_tt_transformers_attention(self):
        hit = lookup("qwen3", "Qwen3Attention")
        assert hit is not None
        assert hit.status == ADAPT
        assert hit.tt_path == "models/tt_transformers/tt/attention.py"
        assert hit.tt_class == "Attention"

    def test_qwen3_rotary_is_adapt(self):
        hit = lookup("qwen3", "Qwen3RotaryEmbedding")
        assert hit is not None
        assert hit.status == ADAPT
        assert hit.tt_path == "models/tt_transformers/tt/rope.py"

    def test_unknown_class_returns_none(self):
        assert lookup("qwen3", "NotARealClass") is None

    def test_wrong_model_type_still_matches_wildcard_block(self):
        hit = lookup("llama", "Qwen3RMSNorm")
        assert hit is not None
        assert hit.tt_class == "RMSNorm"
        assert hit.model_types == (), "RMSNorm BuildingBlock is wildcard (all non-SSM models)"

    def test_unknown_class_with_known_model_type_returns_none(self):
        assert lookup("qwen3", "FoobarWidget") is None

    def test_phi4_attention_shares_with_qwen3(self):
        hit = lookup("phi4", "Phi4Attention")
        assert hit is not None
        assert hit.tt_path == "models/tt_transformers/tt/attention.py"

    def test_model_type_case_insensitive(self):
        assert lookup("QWEN3", "Qwen3RMSNorm") is not None
        assert lookup("Qwen3", "Qwen3RMSNorm") is not None

    def test_empty_class_returns_none(self):
        assert lookup("qwen3", None) is None
        assert lookup("qwen3", "") is None

    def test_none_model_type_matches_wildcard_blocks(self):
        hit = lookup(None, "Qwen3RMSNorm")
        assert hit is not None
        assert hit.tt_class == "RMSNorm"

    def test_qwen3_has_multiple_entries(self):
        entries = entries_for_model_type("qwen3")
        assert len(entries) >= 5

    def test_every_entry_points_at_existing_file_in_repo_layout(self):
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[3]
        for entry in all_entries():
            if "auto-derived from upstream" in (entry.notes or ""):
                continue  # overlay entries point at upstream paths, not guaranteed in the local checkout
            target = repo_root / entry.tt_path
            assert target.is_file(), f"Registry entry {entry.concept} points at non-existent " f"file {entry.tt_path}"


class TestConceptLookup:
    def test_attention_concept_for_qwen3(self):
        hit = lookup_by_concept("qwen3", "attention")
        assert hit is not None and hit.tt_class == "Attention"

    def test_self_attention_concept_falls_back_to_attention(self):
        hit = lookup_by_concept("qwen3", "self_attention")
        assert hit is not None and hit.tt_class == "Attention"

    def test_mlp_concept(self):
        hit = lookup_by_concept("llama", "mlp")
        assert hit is not None and hit.tt_class == "MLP"

    def test_rope_concept(self):
        hit = lookup_by_concept("qwen3", "rope")
        assert hit is not None and hit.tt_class == "RotaryEmbedding"

    def test_unknown_concept_returns_none(self):
        assert lookup_by_concept("qwen3", "definitely_not_a_concept") is None


class TestBuildingBlocksDerivation:
    def test_derived_entries_include_attention(self):
        derived = _derived_entries()
        assert any(e.tt_class == "Attention" for e in derived), (
            "BUILDING_BLOCKS derivation must produce an Attention entry " "(GQA attention block)"
        )

    def test_derived_entries_include_rmsnorm(self):
        derived = _derived_entries()
        rms = [e for e in derived if e.tt_class == "RMSNorm"]
        assert rms, "BUILDING_BLOCKS derivation must produce an RMSNorm entry"
        assert any(e.tt_path == "models/common/rmsnorm.py" for e in rms), (
            "RMSNorm entry must use registry_tt_path=models/common/rmsnorm.py, "
            "not the human-readable 'wraps ...' string"
        )

    def test_derived_entries_skip_human_readable_paths(self):
        derived = _derived_entries()
        for e in derived:
            assert " (wraps " not in e.tt_path
            assert " + " not in e.tt_path
            assert "{" not in e.tt_path


@pytest.mark.network
class TestModuleTreeIntegration:
    def test_qwen3_embedding_classifies_with_zero_new_components(self):
        try:
            comps = _extract_components_from_module_tree(
                new_model_id="Qwen/Qwen3-Embedding-8B",
                new_model_type="qwen3",
            )
        except Exception as exc:
            pytest.skip(f"HF model load failed (offline?): {exc}")
        if not comps:
            pytest.skip("module-tree returned no components (HF load failed)")
        counts = {REUSE: 0, ADAPT: 0, NEW: 0}
        for c in comps:
            counts[c.status] = counts.get(c.status, 0) + 1
        assert counts[NEW] == 0, (
            f"After registry bridge, Qwen3-Embedding-8B should have 0 NEW "
            f"components. Got {counts}. "
            f"Components: {[(c.name, c.class_name, c.status) for c in comps]}"
        )
        assert (
            counts[ADAPT] >= 1
        ), f"At least RMSNorm / RotaryEmbedding should be ADAPT (never trusted blind). Got {counts}"
