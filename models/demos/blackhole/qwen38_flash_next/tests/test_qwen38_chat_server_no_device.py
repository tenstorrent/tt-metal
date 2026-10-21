# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The chat server's start-up and request checks without a device: the shipped greedy acceptance records and their
loader's refusals, the MTP switches the server does not read, and the refused ``mtp_drafts`` request field."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_server as server
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_protocol import Qwen38ChatRequestRejected

ACCEPTANCE_RECORDS = Path(server.__file__).parent / "acceptance" / "greedy-prompts.jsonl"
RECORD_FIELDS = {"prompt", "prompt_token_ids", "generated_token_ids", "stop_reason", "sha256"}


def _record(**overrides) -> dict:
    record = {
        "prompt": "json",
        "mode": "greedy",
        "prompt_length": 3,
        "prompt_token_ids": [1, 2, 3],
        "generated_token_ids": [4, 5],
        "stop_reason": "eos",
        "sha256": "0" * 64,
    }
    record.update(overrides)
    return record


def test_the_shipped_acceptance_records_load_with_the_gate_prompt_first() -> None:
    records = server.load_acceptance_records(ACCEPTANCE_RECORDS)
    names = [record["prompt"] for record in records]
    assert len(records) == 13 and len(set(names)) == 13, f"records {names}"
    assert names[0] == server.ACCEPTANCE_GATE_PROMPT and names[1:] == sorted(names[1:]), f"order {names}"
    for record in records:
        assert set(record) == RECORD_FIELDS, f"{record['prompt']}: fields {sorted(record)}"
        digest = record["sha256"]
        assert len(digest) == 64 and set(digest) <= set("0123456789abcdef"), f"{record['prompt']}: sha256 {digest!r}"
        assert record["prompt_token_ids"] and record["generated_token_ids"], f"{record['prompt']}: empty token list"


def test_the_loader_names_the_line_it_refuses(tmp_path, expect_error) -> None:
    path = tmp_path / "records.jsonl"
    good = json.dumps(_record())
    for second, message in (
        ("{", "records.jsonl:2 is not JSON"),
        ("[1, 2]", "records.jsonl:2 is not a JSON object, got list"),
        (json.dumps(_record(sha256="A" * 64)), "records.jsonl:2 is not a greedy prompt record"),
        (json.dumps(_record(sha256="0" * 63)), "records.jsonl:2 is not a greedy prompt record"),
        (json.dumps(_record(prompt_length=4)), "records.jsonl:2 is not a greedy prompt record"),
        (json.dumps(_record(mode="sampled")), "records.jsonl:2 is not a greedy prompt record"),
    ):
        path.write_text(f"{good}\n{second}\n", encoding="utf-8")
        with expect_error(ValueError, match=message):
            server.load_acceptance_records(path)
    path.write_text("", encoding="utf-8")
    with expect_error(ValueError, match="no greedy prompt records"):
        server.load_acceptance_records(path)


@pytest.mark.parametrize("name", sorted(server.UNREAD_MTP_SWITCHES))
@pytest.mark.parametrize("value", ["0", "1", ""])
def test_an_unread_mtp_switch_refuses_the_start_whatever_its_value(expect_error, name, value) -> None:
    with expect_error(SystemExit, match=f"{name} is not read by this server \\(.+\\): unset it"):
        server.refuse_unread_switches({name: value})


def test_the_start_takes_an_environment_without_unread_switches(expect_error) -> None:
    server.refuse_unread_switches({"QWEN38_MTP_DEVICE_ACCEPT": "1", "QWEN38_FUSED_OFF": "gr_write"})  # served switches
    first, second = sorted(server.UNREAD_MTP_SWITCHES)[:2]
    with expect_error(SystemExit, match=f"{first} is not read .+; {second} is not read "):
        server.refuse_unread_switches({"QWEN38_FUSED_OFF": "all", first: "1", second: "0"})


@pytest.mark.parametrize("where", ["top", "extra_body"])
@pytest.mark.parametrize("drafts", [4, 5])
def test_a_request_naming_mtp_drafts_is_refused_with_its_param(where, drafts) -> None:
    messages = [{"role": "user", "content": "hi"}]
    document = {
        "messages": messages,
        **({"mtp_drafts": drafts} if where == "top" else {"extra_body": {"mtp_drafts": drafts}}),
    }
    with pytest.raises(Qwen38ChatRequestRejected) as info:  # allow-pytest.raises: inspect the captured exception object
        server.parse_chat_request(document, seed=1)
    assert (info.value.param, info.value.code) == ("mtp_drafts", "bad_request"), f"{info.value.param} {info.value.code}"
    assert f"one drafting chain): drop it, got {drafts}" in str(info.value), str(info.value)
    parsed = server.parse_chat_request({"messages": messages, "mtp_drafts": None}, seed=1)  # JSON null: absent
    assert "mtp_drafts" not in parsed, sorted(parsed)
