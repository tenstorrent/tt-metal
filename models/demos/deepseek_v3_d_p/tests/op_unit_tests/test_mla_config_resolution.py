# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for ``ttMLA``'s tuned-config resolution: gating tags and multi-candidate selection.

``MLA_MATMUL_CONFIG[weight][seq_len_local]`` and ``MLA_SDPA_CONFIG[seq_len_local]`` are shared across
every model variant, so a single ``seq_len_local`` slot can be wanted by more than one architecture --
Kimi-K2.6 (64 heads) and Kimi-K3 (96 heads) both run chunked prefill at ``640``. The declared tags
(``num_heads`` / ``q_lora_rank`` / ``chunked_only`` / ``dense_head_cap_non_dsa``) only *reject* a
candidate; picking between alternatives is what ``_select_cfg`` adds.

The failure this guards against is silent and expensive: a variant picking up another variant's
program config produces a dimensionally-invalid tiling (grid overflow, wrong per_core_N) rather than
a clean fallback to untuned defaults.

No device: ``_cfg_matches`` / ``_select_cfg`` depend only on ``num_heads``, ``q_lora_rank``,
``is_chunked`` and ``_is_dsa_family``, so the tests drive them on a bare instance.
"""

import pytest

from models.demos.deepseek_v3_d_p.tt.mla.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.mla_config import MLA_MATMUL_CONFIG, MLA_SDPA_CONFIG

# Sentinels so a returned config is identifiable without constructing real program configs.
K26 = {"num_heads": 64, "q_lora_rank": 1536, "chunked_only": True, "tag": "k26"}
K3 = {"num_heads": 96, "q_lora_rank": 1536, "chunked_only": True, "tag": "k3"}
UNTAGGED = {"tag": "untagged"}


def _mla(num_heads, q_lora_rank=1536, is_chunked=True, is_dsa=False):
    """A ttMLA with only the attributes the resolvers read. Bypasses __init__ (which needs a device)."""
    mla = object.__new__(ttMLA)
    mla.num_heads = num_heads
    mla.q_lora_rank = q_lora_rank
    mla.is_chunked = is_chunked
    mla._is_dsa_family = is_dsa
    return mla


@pytest.mark.parametrize(
    "num_heads,expected",
    [(64, "k26"), (96, "k3")],
    ids=["kimi_k2_6", "kimi_k3"],
)
def test_select_cfg_picks_the_matching_head_count(num_heads, expected):
    """Two candidates in one slot: each variant must get its own, not the other's."""
    assert _mla(num_heads)._select_cfg([K26, K3])["tag"] == expected
    # Order must not matter when the tags are mutually exclusive.
    assert _mla(num_heads)._select_cfg([K3, K26])["tag"] == expected


def test_select_cfg_returns_none_when_no_candidate_matches():
    """128 heads matches neither; the caller must fall back to untuned defaults, not guess."""
    assert _mla(128)._select_cfg([K26, K3]) is None


def test_select_cfg_accepts_a_bare_dict():
    """Backwards compatibility: every existing entry is a single dict, not a list."""
    assert _mla(64)._select_cfg(K26)["tag"] == "k26"
    assert _mla(96)._select_cfg(K26) is None
    assert _mla(None)._select_cfg(None) is None


def test_select_cfg_is_first_match_wins():
    """With overlapping candidates the earlier one wins, so list order is priority order."""
    specific = {"num_heads": 96, "tag": "specific"}
    catch_all = {"tag": "catch_all"}
    assert _mla(96)._select_cfg([specific, catch_all])["tag"] == "specific"
    assert _mla(96)._select_cfg([catch_all, specific])["tag"] == "catch_all"
    # An untagged catch-all still matches a head count no specific candidate covers.
    assert _mla(128)._select_cfg([specific, catch_all])["tag"] == "catch_all"


def test_q_lora_rank_tag_separates_same_head_count():
    """GLM-5.1 and Kimi-K2.6 are both 64 heads but differ in q_lora_rank (2048 vs 1536)."""
    assert _mla(64, q_lora_rank=1536)._select_cfg([K26])["tag"] == "k26"
    assert _mla(64, q_lora_rank=2048)._select_cfg([K26]) is None


def test_chunked_only_tag_is_per_instance():
    """The 640 set is only dimensionally valid in chunked mode."""
    assert _mla(96, is_chunked=True)._select_cfg([K3])["tag"] == "k3"
    assert _mla(96, is_chunked=False)._select_cfg([K3]) is None
    assert _mla(96, is_chunked=False)._select_cfg([UNTAGGED])["tag"] == "untagged"


def test_dense_head_cap_applies_to_non_dsa_only():
    """dense_head_cap_non_dsa rejects above-cap non-DSA models; DSA-family models are exempt."""
    capped = {"dense_head_cap_non_dsa": 64, "tag": "capped"}
    assert _mla(64, is_dsa=False)._select_cfg([capped])["tag"] == "capped"  # at the cap
    assert _mla(96, is_dsa=False)._select_cfg([capped]) is None  # above, non-DSA
    assert _mla(128, is_dsa=True)._select_cfg([capped])["tag"] == "capped"  # above, DSA-exempt


def test_dense_head_cap_is_honoured_by_matmul_resolution_too():
    """The tag is shared, so it cannot be respected by the SDPA path and ignored by the matmul path.

    No shipped matmul config declares it today; this pins the contract so adding one behaves.
    """
    capped = {"dense_head_cap_non_dsa": 64, "tag": "capped"}
    mla = _mla(96)
    assert not mla._cfg_matches(capped)


def test_shipped_configs_resolve_for_every_seq_len_they_declare():
    """Every shipped entry must resolve for at least one plausible model, in whatever form it takes.

    Catches a candidate list whose tags are mutually contradictory (e.g. a num_heads that no shipped
    variant has), which would leave a tuned config permanently dead. Head counts cover the shipped
    dense MLA variants: Kimi-K2.6/2.7 (64), GLM (64), Kimi-K3 (96), DeepSeek-V3.x (128).
    """
    plausible = [
        _mla(64, q_lora_rank=1536),
        _mla(64, q_lora_rank=2048),
        _mla(96, q_lora_rank=1536),
        _mla(128, q_lora_rank=1536),
        _mla(128, q_lora_rank=1536, is_dsa=True),
    ]
    dead = []
    for weight, by_seq in MLA_MATMUL_CONFIG.items():
        for seq_len, entry in by_seq.items():
            candidates = entry if isinstance(entry, (list, tuple)) else [entry]
            for i, cfg in enumerate(candidates):
                if not any(mla._cfg_matches(cfg) for mla in plausible):
                    dead.append(f"MLA_MATMUL_CONFIG[{weight!r}][{seq_len}] candidate {i}")
    for seq_len, entry in MLA_SDPA_CONFIG.items():
        candidates = entry if isinstance(entry, (list, tuple)) else [entry]
        for i, cfg in enumerate(candidates):
            if not any(mla._cfg_matches(cfg) for mla in plausible):
                dead.append(f"MLA_SDPA_CONFIG[{seq_len}] candidate {i}")
    assert not dead, "tuned configs no shipped variant can ever select:\n  " + "\n  ".join(dead)
