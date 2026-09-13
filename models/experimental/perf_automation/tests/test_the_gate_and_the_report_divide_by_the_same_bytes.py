# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The stop gate's ceiling and the roofline table's must describe the same read set.

The report prices each stage by the subtree it streams -- a decode token reads the language backbone
and never the audio encoder -- while _select_perf_target handed compute_target the WHOLE model. On a
two-tower model that ceiling is too LOW, so the measured rate looks closer to it than it is: the run
can be declared at-the-floor while the report, dividing correctly, still shows headroom. Voxtral's
backbone is ~86% of resident, so the gate's ceiling was ~14% pessimistic.

MEASURED ONLY. The share comes from device_section_bytes -- the census walking the BUILT model -- and
never from the checkpoint's ratio, which states disk precision. A wrong ceiling in the stop gate is
worse than a conservative one, so anything unestablished falls back to the whole model.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))

_MF = {
    "device_weight_bytes": 1_000_000,
    "device_section_bytes": {"audio_tower": 140_000, "language_model": 860_000},
    "stage_roots": {"encode": "audio_tower", "prefill": "language_model", "decode": "language_model"},
}


def _share(mf, monkeypatch, isl=None):
    import cc_optimize.perf_mcp as PM

    monkeypatch.setattr(PM, "read_stage_isl_map", lambda *a, **k: (isl if isl is not None else {"decode": 1}))
    monkeypatch.delenv("PERF_MCP_RECURRING_STAGE", raising=False)
    return PM._recurring_subtree_share(mf)


def test_the_recurring_stage_is_priced_from_its_own_subtree(monkeypatch):
    assert abs(_share(_MF, monkeypatch) - 0.86) < 1e-9


def test_a_single_tower_model_is_unchanged(monkeypatch):
    mf = {
        "device_weight_bytes": 1_000_000,
        "device_section_bytes": {"model": 1_000_000},
        "stage_roots": {"decode": "model"},
    }
    assert _share(mf, monkeypatch) == 1.0


def test_no_census_split_means_the_whole_model(monkeypatch):
    """The checkpoint ratio is NOT accepted here: the gate acts on the answer."""
    mf = dict(_MF)
    mf.pop("device_section_bytes")
    assert _share(mf, monkeypatch) == 1.0


def test_no_mapping_means_the_whole_model(monkeypatch):
    mf = dict(_MF)
    mf.pop("stage_roots")
    assert _share(mf, monkeypatch) == 1.0


def test_an_ambiguous_recurring_stage_means_the_whole_model(monkeypatch):
    """Two stages retiring one item each is not an answer about which the headline counts."""
    assert _share(_MF, monkeypatch, isl={"decode": 1, "encode": 1}) == 1.0


def test_a_share_above_one_is_refused(monkeypatch):
    """Total and split from different walks; publishing it would RAISE the ceiling on no evidence."""
    mf = dict(_MF, device_section_bytes={"language_model": 2_000_000})
    assert _share(mf, monkeypatch) == 1.0


def test_the_gate_scales_the_ceiling_it_asks_for():
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    i = src.index("def _select_perf_target(")
    body = src[i : src.index("\ndef ", i + 1)]
    assert "_recurring_subtree_share(mf)" in body, "the gate still divides by the whole model"
    assert "device_weight_bytes=int(" in body, "the scaled byte count never reaches compute_target"


# --------------------------------------------- and the report says WHICH split priced the row


def test_the_share_carries_where_it_came_from(monkeypatch):
    """A ceiling built on the checkpoint's split can read >100% of peak, which is indistinguishable
    from a fast model unless the report says which divisor it used. Voxtral's backbone is 85.8% of
    its checkpoint; its decode measurement implies 77.9% of what is resident."""
    import cc_optimize.summary as S

    monkeypatch.setattr(S, "_SECTION_BYTES", {"audio_tower": 1_328_000_000, "language_model": 8_028_000_000})
    common = {"stage_roots": {"decode": "language_model"}, "blocks": {"a": {}, "b": {}}}

    v, basis = S._share_and_basis(dict(common, device_weight_bytes=1_718_081_696), "decode")
    assert basis == "checkpoint" and abs(v - 0.858) < 0.01

    v, basis = S._share_and_basis(
        dict(common, device_weight_bytes=1_718_081_696, device_section_bytes={"language_model": 1_338_000_000}),
        "decode",
    )
    assert basis == "measured" and abs(v - 0.779) < 0.01

    assert S._share_and_basis({"blocks": {"a": {}, "b": {}}}, "decode")[1] == "refused"
    assert S._share_and_basis({"blocks": {"m": {}}}, "decode")[1] == "whole-model"


def test_a_checkpoint_priced_row_says_so_and_a_measured_one_does_not(monkeypatch):
    import cc_optimize.summary as S
    import cc_optimize.perf_mcp as PM

    monkeypatch.setattr(S, "_SECTION_BYTES", {"audio_tower": 1_328_000_000, "language_model": 8_028_000_000})
    monkeypatch.setattr(S, "_prompt_tokens", lambda: 128)
    monkeypatch.setattr(S, "_request_batch", lambda: 8)
    monkeypatch.setattr(PM, "read_stage_isl_map", lambda *a, **k: {"decode": 1})
    monkeypatch.setattr(PM, "read_stage_isl_per_request_map", lambda *a, **k: {})
    base = {
        "device_weight_bytes": 1_718_081_696,
        "dominant_dtype": "bfloat16",
        "stage_roots": {"decode": "language_model"},
        "blocks": {"audio_tower": {"layers": 32}, "language_model": {"layers": 30, "params": 4_014_000_000}},
    }

    def _render(mf):
        monkeypatch.setattr(S, "_model_facts", lambda: mf)
        return "\n".join(
            S._roofline_tables(
                unit="tok/s/u",
                theo=42.7,
                band=[25.6, 34.1],
                measured=29.5,
                bw_gbps=354.0,
                peak_bw_gbps=512.0,
                active_bytes=1_718_081_696,
                per_unit_ms=2.86,
                profile={"buckets": []},
                stage_ms={"decode": 2.86},
            )
        )

    assert "CHECKPOINT's tower split" in _render(base)
    measured = dict(base, device_section_bytes={"language_model": 1_338_000_000})
    assert "CHECKPOINT's tower split" not in _render(measured)
