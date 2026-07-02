# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""shard_plan is GUIDANCE, not a deterministic shard prescription: the LLM agent derives the scheme,
the gate validates it by gathered-PCC. These tests pin the contract — every weight-bearing compute
layer is shard-eligible and gets principles + references; pure replicate-only roles get None; and
NOTHING here dictates a per-weight axis (that is the agent's job)."""
from scripts.tt_hw_planner import shard_plan
from scripts.tt_hw_planner.shard_plan import is_shard_eligible, shard_guidance


def test_compute_layers_are_eligible_and_get_guidance():
    for name in (
        "self_attn",
        "NemotronHMLP",
        "lm_head",
        "nemotron_h_mamba2_mixer",
        "nemotron_h_m_o_e",
        "nemotron_h_topk_router",
        "some_novel_block",
    ):
        g = shard_guidance(name)
        assert g is not None, name
        assert is_shard_eligible(name) is True
        assert "principles" in g and "reference_hints" in g


def test_replicate_only_roles_return_none():
    for name in ("input_layernorm", "rmsnorm", "embedding", "embed_tokens", "rotary_emb", "act_fn", "dropout"):
        assert shard_guidance(name) is None
        assert is_shard_eligible(name) is False


def test_guidance_covers_the_general_principles():
    g = shard_guidance("attention")
    text = g["principles"].lower()
    for kw in ("column", "row", "all_reduce", "expert", "mamba", "all_gather"):
        assert kw in text


def test_references_point_at_tt_transformers():
    g = shard_guidance("some_block")
    hints = g["reference_hints"]
    assert "tt_transformers" in hints
    assert "attention.py" in hints and "mixtral_moe.py" in hints


def test_no_prescriptive_axis_api():
    assert not hasattr(shard_plan, "shard_spec")
    assert not hasattr(shard_plan, "ShardSpec")
