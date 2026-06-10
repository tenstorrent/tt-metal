"""Tests for lib/kb.py — orchestrator -> opt_transfer KB bridge."""

from __future__ import annotations

import json

from skills.orchestrator.lib.kb import kb_entries_for_block


def _write_entry(root, entry_id, **overrides):
    d = {
        "id": entry_id,
        "fused_op": "ttnn.transformer.attention_softmax",
        "category": "attention",
        "pattern_kind": "chain",
        "torch_pattern": ["matmul", "softmax"],
        "signature": {},
        "config_template": {},
        "weight_transform": None,
        "source": "test",
        "usage_examples": [],
        "applicability_notes": "",
        "status": "in_use",
        "accumulation_sensitive": False,
        "pattern_source": "golden",
        "confidence": "high",
        "unit_test_refs": [],
        "placement_observations": [],
    }
    d.update(overrides)
    (root / f"{entry_id}.json").write_text(json.dumps(d))


def test_missing_dir_returns_empty(tmp_path):
    assert kb_entries_for_block("Attention", "attention", kb_dir=tmp_path / "nope") == []


def test_empty_store_returns_empty(tmp_path):
    assert kb_entries_for_block("Attention", "attention", kb_dir=tmp_path) == []


def test_matching_entry_is_returned(tmp_path):
    _write_entry(tmp_path, "attention_softmax_fuse")
    out = kb_entries_for_block("Attention", "attention", kb_dir=tmp_path)
    assert [e["id"] for e in out] == ["attention_softmax_fuse"]


def test_unrelated_entry_is_dropped(tmp_path):
    _write_entry(tmp_path, "conv_bias_fold", category="convolution", fused_op="ttnn.conv2d", torch_pattern=["conv2d"])
    assert kb_entries_for_block("Attention", "attention", kb_dir=tmp_path) == []


def test_ranking_prefers_more_overlap_and_high_confidence(tmp_path):
    _write_entry(tmp_path, "attn_low", confidence="low")
    _write_entry(tmp_path, "attn_high", confidence="high")
    _write_entry(tmp_path, "mlp_silu", category="mlp", fused_op="ttnn.silu", torch_pattern=["silu", "mul"])
    out = kb_entries_for_block("Attention", "attention", kb_dir=tmp_path)
    ids = [e["id"] for e in out]
    assert "mlp_silu" not in ids
    assert ids.index("attn_high") < ids.index("attn_low")


def test_max_entries_cap(tmp_path):
    for i in range(12):
        _write_entry(tmp_path, f"attention_fuse_{i:02d}")
    out = kb_entries_for_block("Attention", "attention", kb_dir=tmp_path, max_entries=8)
    assert len(out) == 8
