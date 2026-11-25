# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The chat server's start-up and request checks without a device: the shipped greedy acceptance records and their
loader's refusals, the MTP switches the server does not read, the refused ``mtp_drafts`` request field, and the
runtime identity of a tree without git history."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.demos.blackhole.qwen38_flash_next.tests.helpers.chat import (
    chat_request_rejection,
    skip_without_git_checkout,
)
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_server as server
from models.demos.blackhole.qwen38_flash_next.tools import runtime_admission

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
    rejected = chat_request_rejection(server.parse_chat_request, document, seed=1)
    assert (rejected.param, rejected.code) == ("mtp_drafts", "bad_request"), f"{rejected.param} {rejected.code}"
    assert f"one drafting chain): drop it, got {drafts}" in str(rejected), str(rejected)
    parsed = server.parse_chat_request({"messages": messages, "mtp_drafts": None}, seed=1)  # JSON null: absent
    assert "mtp_drafts" not in parsed, sorted(parsed)


def test_a_tree_without_git_history_is_admitted_only_with_its_head_declared(
    expect_error, tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv(runtime_admission.DECLARED_HEAD_VARIABLE, raising=False)
    with expect_error(runtime_admission.RuntimeAdmissionError, match="without git history"):
        runtime_admission.git_identity(tmp_path)
    monkeypatch.setenv(runtime_admission.DECLARED_HEAD_VARIABLE, "abc")
    with expect_error(runtime_admission.RuntimeAdmissionError, match="must be lowercase 40-hex"):
        runtime_admission.git_identity(tmp_path)
    head = "78ae83c92c420ae1cd4aa52b5a4b8217961531cb"
    monkeypatch.setenv(runtime_admission.DECLARED_HEAD_VARIABLE, head)
    identity = runtime_admission.git_identity(tmp_path)
    assert identity == {
        "repo": str(tmp_path),
        "head": head,
        "tree": None,
        "dirty": False,
        "source": "declared",
    }, identity


def test_a_checkout_with_history_refuses_a_declared_head_that_differs(expect_error, monkeypatch) -> None:
    root = runtime_admission.REPO_ROOT
    skip_without_git_checkout(root)
    monkeypatch.delenv(runtime_admission.DECLARED_HEAD_VARIABLE, raising=False)
    identity = runtime_admission.git_identity(root)
    assert "source" not in identity and len(identity["head"]) == 40, identity
    monkeypatch.setenv(runtime_admission.DECLARED_HEAD_VARIABLE, identity["head"])
    declared = runtime_admission.git_identity(root)
    assert declared["head"] == identity["head"], (declared, identity)
    monkeypatch.setenv(runtime_admission.DECLARED_HEAD_VARIABLE, "0" * 40)
    with expect_error(runtime_admission.RuntimeAdmissionError, match="differs from the checkout head"):
        runtime_admission.git_identity(root)
