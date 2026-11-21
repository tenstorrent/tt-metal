# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A published card's run section must work when pasted, and its Tenstorrent section must name the
hardware its numbers were measured on.

minicpm5_2b, 2026-11-10: the card said "Hardware: Tenstorrent p150" while every figure came from 1 chip
of a p300x2; its `uv tool install tenstorrent` gave a `tt` that crashed (typer 0.29.0, released the day
before, broke tenstorrent 1.2.1); its Python example needed `openai` without saying so; it claimed a
Tenstorrent column in benchmark tables that had none; and it never said the package serves at most
3,072 tokens per request, while the upstream card below it advertises long context.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tt_hw_planner.commands import publish_hf as ph  # noqa: E402

SERVED = {
    "max_num_seqs": 32,
    "max_model_len": 3072,
    "measured_on": "1 Tenstorrent Blackhole chip of a p300x2",
    "rows": [(1, 128, 128, 0.04, 10.9, 91.5, 91), (32, 128, 128, 0.5, 11.9, 84.2, 2693)],
}


def _card(monkeypatch, upstream_body):
    monkeypatch.setattr(ph, "_fetch_upstream_card", lambda base, token: upstream_body)
    state = {"batch": 32, "throughput": {"current": 99.25}, "served": SERVED, "env": {"arch": "blackhole"}}
    return ph._compose_mirrored_card(state, "minicpm5_2b", "openbmb/MiniCPM5-2B", "381552b92d", None, None, "o/r")


def test_the_card_names_the_chip_its_numbers_were_measured_on_beside_the_target():
    block = ph._tt_validated_block({"served": SERVED}, "Tenstorrent p150", None)
    assert "| Hardware | Tenstorrent p150 |" in block
    assert "| Measured on | 1 Tenstorrent Blackhole chip of a p300x2 |" in block
    assert "measured on 1 Tenstorrent Blackhole chip of a p300x2 through" in block


def test_a_package_measured_on_its_target_gets_one_hardware_row():
    block = ph._tt_validated_block({"served": {**SERVED, "measured_on": "Tenstorrent p150"}}, "Tenstorrent p150", None)
    assert "| Measured on |" not in block


def test_the_card_says_how_many_tokens_one_request_may_use():
    block = ph._tt_validated_block({"served": SERVED}, "Tenstorrent p150", None)
    assert "| Context per request (`tt serve`) | 3,072 tokens, prompt + answer |" in block
    no_limit = ph._tt_validated_block({"served": {**SERVED, "max_model_len": None}}, "Tenstorrent p150", None)
    assert "Context per request" not in no_limit


def test_the_run_section_installs_a_tt_that_starts_and_names_the_client_it_needs():
    run = ph._tt_run_section("o/r", "openbmb/MiniCPM5-2B")
    assert "uv tool install tenstorrent --with 'typer<0.29'" in run
    assert "`pip install openai`" in run


def test_the_intro_claims_a_tenstorrent_column_only_when_the_tables_got_one(monkeypatch):
    without = _card(monkeypatch, "## Evaluation\n\nScores are in the chart above.\n")
    assert "Tenstorrent column" not in without
    assert "brought up on 1 Tenstorrent Blackhole chip of a p300x2" in without
    with_table = _card(
        monkeypatch,
        "## Evaluation\n\n| Benchmark | MiniCPM5-2B | Other |\n| --- | --- | --- |\n"
        "| MMLU | 70.1 | 65.0 |\n| GSM8K | 80.2 | 75.3 |\n",
    )
    assert "| Tenstorrent |" in with_table and "Tenstorrent column" in with_table
