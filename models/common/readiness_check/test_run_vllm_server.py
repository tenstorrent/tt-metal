# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from models.common.readiness_check import run_vllm_server
from models.common.readiness_check.run_vllm_server import _qualitative_prompt_mode, _request_qualitative_completion


@pytest.mark.parametrize(
    ("chat_template", "expected"),
    [("{{ messages }}", "chat"), (None, "completion")],
)
def test_qualitative_prompt_mode_follows_checkpoint_chat_template(monkeypatch, chat_template, expected):
    tokenizer_factory = Mock(return_value=SimpleNamespace(chat_template=chat_template))
    monkeypatch.setattr(run_vllm_server.AutoTokenizer, "from_pretrained", tokenizer_factory)

    assert _qualitative_prompt_mode("org/model") == expected
    tokenizer_factory.assert_called_once_with("org/model")


def test_request_qualitative_completion_uses_chat_endpoint_for_chat_models():
    chat_create = Mock(
        return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="chat response"))])
    )
    completion_create = Mock()
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=chat_create)),
        completions=SimpleNamespace(create=completion_create),
    )

    result = _request_qualitative_completion(
        client=client,
        hf_model="org/instruct-model",
        prompt="Question",
        prompt_mode="chat",
        temperature=0.0,
    )

    assert result == "chat response"
    chat_create.assert_called_once_with(
        messages=[{"role": "user", "content": "Question"}],
        model="org/instruct-model",
        max_tokens=256,
        temperature=0.0,
    )
    completion_create.assert_not_called()


def test_request_qualitative_completion_uses_raw_endpoint_for_base_models():
    completion_create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(text="base response")]))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=Mock())),
        completions=SimpleNamespace(create=completion_create),
    )

    result = _request_qualitative_completion(
        client=client,
        hf_model="org/base-model",
        prompt="Continue",
        prompt_mode="completion",
        temperature=0.7,
        top_p=0.9,
    )

    assert result == "base response"
    completion_create.assert_called_once_with(
        prompt="Continue",
        model="org/base-model",
        max_tokens=256,
        temperature=0.7,
        top_p=0.9,
    )
