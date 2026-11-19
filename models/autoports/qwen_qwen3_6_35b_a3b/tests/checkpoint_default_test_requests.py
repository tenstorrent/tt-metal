# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Explicit request configuration for short-answer canonical chat assertions.

Load only with pytest -p models.autoports.qwen_qwen3_6_35b_a3b.tests.
checkpoint_default_test_requests. The server keeps checkpoint thinking defaults.
This supplemental validation adds the existing HTTP request override, preserving
the canonical tests' assertions, caps, seeds, constraints, and concurrency.
It is not an unchanged-default canonical-suite result or a shared plugin option.
Only the six named short-answer tests receive this explicit request mode.
Recall tests use the plugin's existing chat recall option; their 24-token cap,
passphrase assertions and concurrent scheduling workload remain unchanged.
"""

import json
import os
from pathlib import Path

import pytest
from openai.resources.chat.completions import AsyncCompletions


@pytest.fixture(autouse=True)
def explicit_nonthinking_chat_requests(monkeypatch, request):
    selected = {
        "test_host_only_params.py::TestHostOnlyParameters::test_bad_words",
        "test_structured_output_dp1.py::test_dp1_full_capacity_mixes_structured_and_plain_requests",
        "test_structured_output_dp1.py::test_dp1_reuses_slots_with_mixed_requests",
        "test_chunked_prefill.py::test_a_solo_split_prefill_recalls_its_needle",
        "test_chunked_prefill.py::test_a_long_split_prefill_recalls_its_needle",
        "test_chunked_prefill.py::test_prefills_sharing_a_step_each_recall_their_own_needle",
    }
    if request.node.nodeid.split("/")[-1].split("[")[0] not in selected:
        return
    original = AsyncCompletions.create
    evidence = Path(os.environ["QWEN_CHAT_TEST_REQUEST_LOG"])

    async def create(client, *args, **kwargs):
        body = dict(kwargs.get("extra_body") or {})
        assert "chat_template_kwargs" not in body, "Do not replace an explicit request setting"
        body["chat_template_kwargs"] = {"enable_thinking": False}
        kwargs["extra_body"] = body
        row = {"node": request.node.nodeid, "request": kwargs}
        response = await original(client, *args, **kwargs)
        row["response"] = response.model_dump(mode="json")
        with evidence.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        return response

    monkeypatch.setattr(AsyncCompletions, "create", create)
