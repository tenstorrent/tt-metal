# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the architecture-donor finder. No device required.

    pytest models/tooling/arch_donor/tests/test_arch_donor.py

The parameter estimate is the load-bearing test: producing the right total
requires having read every MoE field alias correctly, so a misread expert count
fails by 10-100x rather than by a plausible margin.
"""

import os

import pytest

from models.tooling.arch_donor import compare as CMP
from models.tooling.arch_donor import corpus as CO
from models.tooling.arch_donor import signature as S

REPO = CO.REPO
CONFIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")

# Published totals from the vendors' own model cards.
#
# The "activated parameter" figure is asserted as a BRACKET because vendors
# disagree on whether embeddings count: MiniMax's 23B excludes them entirely,
# while OpenAI's 5.1B/3.6B are exactly our figure plus one embedding matrix. So
# a published number must land inside [active_no_embed_B, active_embed_once_B],
# with 3% slack for their rounding.
TOL_TOTAL = 0.05
BRACKET_SLACK = 0.03

PARAM_TRUTH = [
    ("models/demos/minimax_m3/configs/MiniMax-M3/config.json", "MiniMax-M3", 428, 23),
    ("models/demos/gpt_oss/configs/gpt-oss-120b/config.json", "gpt-oss-120b", 116.8, 5.1),
    ("models/demos/gpt_oss/configs/gpt-oss-20b/config.json", "gpt-oss-20b", 20.9, 3.6),
    ("models/demos/deepseek_v3/reference/config.json", "DeepSeek-V3", 671, 37),
    (
        "models/demos/llama3_70b_galaxy/model_params/Llama-3.3-70B-Instruct/config.json",
        "Llama-3.3-70B",
        70.6,
        68.9,
    ),
    ("models/tt_transformers/model_params/Mistral-7B-Instruct-v0.3/config.json", "Mistral-7B", 7.25, 7.0),
    ("models/tt_transformers/model_params/Qwen3-32B/config.json", "Qwen3-32B", 32.8, 31.0),
]

MECH_TRUTH = {
    "MiniMax-M3": (
        "models/demos/minimax_m3/configs/MiniMax-M3/config.json",
        {
            ("attention", "kind"): "GQA",
            ("attention", "qk_norm"): "per_head",
            ("attention", "rope_coverage"): "partial",
            ("attention", "rope_scaling"): "none",
            ("attention", "sparsity"): "block_sparse",
            ("attention", "qkv_bias"): False,
            ("mlp", "glu"): "swiglu_clamped",
            ("mlp", "moe"): True,
            ("mlp", "router_score"): "sigmoid",
            ("mlp", "router_bias"): True,
            ("mlp", "shared_expert"): True,
            ("mlp", "hybrid_schedule"): True,
            ("norm", "style"): "gemma_1plus",
        },
    ),
    "gpt-oss-120b": (
        "models/demos/gpt_oss/configs/gpt-oss-120b/config.json",
        {
            ("attention", "kind"): "GQA",
            ("attention", "sinks"): True,
            ("attention", "sparsity"): "hybrid_sliding",
            ("attention", "qk_norm"): "none",
            ("mlp", "glu"): "swiglu_clamped",
            ("mlp", "moe"): True,
            ("norm", "style"): "standard",
            ("global", "quant"): "mxfp4",
        },
    ),
    "DeepSeek-V3": (
        "models/demos/deepseek_v3/reference/config.json",
        {
            ("attention", "kind"): "MLA",
            ("attention", "rope_coverage"): "partial",
            ("mlp", "moe"): True,
            ("mlp", "router_score"): "sigmoid",
            ("mlp", "router_bias"): True,
            ("mlp", "shared_expert"): True,
            ("mlp", "hybrid_schedule"): True,
            ("norm", "style"): "standard",
        },
    ),
    "Llama-3.3-70B": (
        "models/demos/llama3_70b_galaxy/model_params/Llama-3.3-70B-Instruct/config.json",
        {
            ("attention", "kind"): "GQA",
            ("attention", "qk_norm"): "none",
            ("attention", "rope_coverage"): "full",
            ("attention", "rope_scaling"): "llama3",
            ("attention", "sparsity"): "dense",
            ("mlp", "glu"): "swiglu_silu",
            ("mlp", "moe"): False,
            ("norm", "style"): "standard",
        },
    ),
}

# Fetched from huggingface.co/mistralai/Mistral-Medium-3.5-128B on 2026-08-25 so
# the retrodiction tests stay hermetic and offline.
MISTRAL_MEDIUM = os.path.join(CONFIGS, "Mistral-Medium-3.5-128B.config.json")


@pytest.fixture(scope="module")
def donors():
    return CO.build_corpus()


@pytest.mark.parametrize("rel,name,exp_total,exp_active", PARAM_TRUTH, ids=[t[1] for t in PARAM_TRUTH])
def test_param_estimate_matches_model_card(rel, name, exp_total, exp_active):
    p = S.from_path(os.path.join(REPO, rel), name=name).params
    assert abs(p["total_B"] - exp_total) <= TOL_TOTAL * exp_total, f"{name} total_B={p['total_B']}"
    lo, hi = p["active_no_embed_B"], p["active_embed_once_B"]
    assert (
        lo * (1 - BRACKET_SLACK) <= exp_active <= hi * (1 + BRACKET_SLACK)
    ), f"{name} published active {exp_active} outside bracket [{lo}, {hi}]"


@pytest.mark.parametrize("name", sorted(MECH_TRUTH), ids=sorted(MECH_TRUTH))
def test_mechanism_signature(name):
    rel, truth = MECH_TRUTH[name]
    sig = S.from_path(os.path.join(REPO, rel), name=name)
    for (block, field), want in truth.items():
        assert sig.mech[block][field] == want, f"{name}.{block}.{field}"


def test_mistral_medium_signature():
    """The target that motivated the tool: dense GQA, YaRN, fp8, no QK-norm."""
    sig = S.from_path(MISTRAL_MEDIUM, name="Mistral-Medium-3.5-128B")
    assert sig.model_type == "ministral3"
    assert sig.mech["attention"] == {
        "kind": "GQA",
        "qk_norm": "none",
        "rope_coverage": "full",
        "rope_scaling": "yarn",
        "sparsity": "dense",
        "sinks": False,
        "out_gate": False,
        "qkv_bias": False,
        "kv_shared_layers": False,
    }
    assert sig.mech["mlp"]["glu"] == "swiglu_silu" and not sig.mech["mlp"]["moe"]
    assert sig.mech["norm"]["style"] == "standard"
    assert sig.mech["global"]["quant"] == "fp8"
    assert abs(sig.params["total_B"] - 125.0) < 5.0


def _top(target, block, donors, galaxy_only=True):
    rows = CMP.rank_for_block(block, target, [d for d in donors if d.name != target.name], galaxy_only)
    assert rows, f"no donors ranked for {block}"
    return rows[0]


def test_retrodict_minimax_moe_came_from_deepseek(donors):
    """MiniMax-M3's MoE was built by extending DeepSeek's
    unified_routed_expert_ffn with a SwiGLU-OAI activation (tt-metal
    a4e461ee4cc). The ranking must rediscover both the donor and the one swap."""
    m3 = S.from_path(os.path.join(REPO, "models/demos/minimax_m3/configs/MiniMax-M3/config.json"), name="MiniMax-M3")
    r = _top(m3, "mlp", donors)
    assert r["donor"].name == "deepseek_v3"
    assert r["verdict"] == "near"
    assert [d["field"] for d in r["diffs"] if d["severity"] == "dataflow"] == ["glu"]


def test_mistral_medium_nearest_donor_is_llama_70b_galaxy(donors):
    """Dense GQA + silu SwiGLU + standard RMSNorm + an identical 28672 FFN makes
    the Galaxy llama the donor, not the MoE/sparse-attention models."""
    mm = S.from_path(MISTRAL_MEDIUM, name="Mistral-Medium-3.5-128B")
    attn, mlp = _top(mm, "attention", donors), _top(mm, "mlp", donors)
    assert attn["donor"].name == "Llama-3.3-70B-Instruct"
    assert attn["verdict"] == "compatible"
    # only the RoPE scaling differs, and it is host-side table generation
    assert [d["field"] for d in attn["diffs"]] == ["rope_scaling"]
    assert attn["diffs"][0]["severity"] == "host"
    assert mlp["donor"].name == "Llama-3.3-70B-Instruct"
    assert mlp["verdict"] == "identical"
    assert mlp["shape_dist"] == pytest.approx(0.0)


def test_target_is_never_compared_against_itself(donors):
    m3 = S.from_path(os.path.join(REPO, "models/demos/minimax_m3/configs/MiniMax-M3/config.json"), name="MiniMax-M3")
    ranked = CMP.rank_for_block("mlp", m3, [d for d in donors if d.name != m3.name], True)
    assert all(r["donor"].name != "MiniMax-M3" for r in ranked)


def test_corpus_tiers_and_pointers(donors):
    by_name = {d.name: d for d in donors}
    assert by_name["Llama-3.3-70B-Instruct"].tier == "proven"
    assert by_name["Llama-3.3-70B-Instruct"].impl_dir == "models/demos/llama3_70b_galaxy"
    # perf CI for these had not landed as of 2026-08; they must not read as proven
    assert by_name["MiniMax-M3"].tier == "in-flight"
    assert by_name["deepseek_v3"].tier == "in-flight"
    assert by_name["Mistral-7B-Instruct-v0.3"].tier == "reference"
    assert by_name["MiniMax-M3"].galaxy_class and not by_name["Mistral-7B-Instruct-v0.3"].galaxy_class


def test_unknown_mechanisms_are_flagged_not_guessed():
    """A model_type with no traits entry must announce itself, so `unverified`
    verdicts are traceable to a hole in the table rather than a silent match."""
    sig = S.build(
        {
            "model_type": "totally_new_arch",
            "num_hidden_layers": 4,
            "hidden_size": 64,
            "num_attention_heads": 8,
            "vocab_size": 100,
        },
        name="synthetic",
    )
    assert any("no traits entry" in n for n in sig.notes)
    assert sig.mech["attention"]["qk_norm"] == S.UNKNOWN
