# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The greedy-only warning (greedy_only.py) under vLLM's real input processor, with the plugin's block-output patch
installed by a real VllmConfig of the served class (CPU, no device): a greedy-only class (the MTP class) logs one
warning the first time a request with temperature > 0 arrives, and the request still reaches the engine with the
plugin's neutralized clone (the caller's params untouched); greedy requests and a sampling class never warn.

usage: PYTHONPATH=<vllm-tt-plugin src>:<tt-metal> python -m pytest --noconftest -c /dev/null \\
    qwen36_resident/tests/test_greedy_only_warning.py   (a venv with vLLM)
"""
from unittest.mock import patch

import pytest
from loguru import logger

pytest.importorskip("vllm_tt_plugin")
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.input_processor import InputProcessor

from qwen36_resident import generator_vllm, greedy_only
from qwen36_resident.tests.test_vllm_rows_engine import vllm_config


@pytest.fixture
def warnings(monkeypatch):
    monkeypatch.setitem(greedy_only._state, "warned", False)
    got = []
    sink = logger.add(lambda m: got.append(m.record["message"]), level="WARNING")
    yield got
    logger.remove(sink)


def serve(cls, temperatures):
    """the API server's input processing of one request per temperature; (caller params, engine requests)"""
    assert greedy_only._state["installed"], "importing generator_vllm installs the hook"
    with patch("vllm.model_executor.model_loader.utils.get_model_architecture", return_value=(cls, None)):
        processor = InputProcessor(vllm_config(1, True, True, cls=cls))
        sent, got = [], []
        for i, t in enumerate(temperatures):
            params = SamplingParams(temperature=t, top_k=20, top_p=0.95, max_tokens=8)
            prompt = {"type": "token", "prompt_token_ids": [5, 6, 7]}
            got.append(processor.process_inputs(f"r{i}", prompt, params, ("generate",)))
            sent.append(params)
    return sent, got


def test_greedy_only_class_warns_once(warnings):
    cls = generator_vllm.Qwen36ResidentMTPForCausalLM
    assert cls.greedy_only
    sent, got = serve(cls, [0.0, 0.7, 1.0, 0.9])
    hits = [w for w in warnings if "greedy-only" in w]
    assert len(hits) == 1 and "request r1 asks for temperature 0.7" in hits[0], warnings
    assert [p.temperature for p in sent] == [0.0, 0.7, 1.0, 0.9], "the caller's params stay as asked"
    # the plugin's block-output clone: temperature 1 for every request (the class masks its prefill logits)
    assert all(r.sampling_params.temperature == 1.0 for r in got)


def test_greedy_requests_do_not_warn(warnings):
    serve(generator_vllm.Qwen36ResidentMTPForCausalLM, [0.0, 0.0])
    assert not [w for w in warnings if "greedy-only" in w]


def test_a_sampling_class_does_not_warn(warnings):
    cls = generator_vllm.Qwen36ResidentForCausalLM
    assert not getattr(cls, "greedy_only", False)
    sent, got = serve(cls, [0.7, 1.0])
    assert not [w for w in warnings if "greedy-only" in w]
    assert [r.sampling_params.temperature for r in got] == [0.7, 1.0], "a sampling class keeps the request's sampling"
