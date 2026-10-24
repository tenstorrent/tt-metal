# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host tests for the HF reference logits cache behind test_full_model (no device, no HF model)."""
import os

import pytest
import torch

from models.demos.gemma4.tests.hf_reference_cache import hf_reference_cache_dir, hf_reference_key, hf_reference_logits


@pytest.fixture
def ids():
    return torch.tensor([[2, 651, 6037, 576, 7001, 603] + [0] * 26], dtype=torch.int64)


def test_cache_dir_precedence(monkeypatch):
    monkeypatch.setenv("GEMMA4_HF_REF_CACHE", "/x/explicit")
    monkeypatch.setenv("TT_CACHE_PATH", "/x/tt")
    assert hf_reference_cache_dir() == "/x/explicit"
    monkeypatch.setenv("GEMMA4_HF_REF_CACHE", "")
    assert hf_reference_cache_dir() == os.path.join("/x/tt", "hf_reference")
    monkeypatch.delenv("TT_CACHE_PATH")
    assert hf_reference_cache_dir() is None


def test_key_is_a_function_of_model_prompt_and_dtype(ids):
    k = hf_reference_key("google/gemma-4-31B-it", ids)
    assert k.startswith("google--gemma-4-31B-it__bfloat16__32__") and k.endswith(".pt")
    assert k == hf_reference_key("google/gemma-4-31B-it/", ids.clone())
    other = ids.clone()
    other[0, 3] = 999
    assert k != hf_reference_key("google/gemma-4-31B-it", other)
    assert k != hf_reference_key("google/gemma-4-12B-it", ids)
    assert k != hf_reference_key("google/gemma-4-31B-it", ids, dtype="float32")


def test_miss_computes_and_hit_loads(tmp_path, monkeypatch, ids):
    monkeypatch.setenv("GEMMA4_HF_REF_CACHE", str(tmp_path / "ref"))
    calls = []

    def compute():
        calls.append(1)
        torch.manual_seed(len(calls))
        return torch.randn(1, 32, 1000)

    a = hf_reference_logits("google/gemma-4-31B-it", ids, compute)
    b = hf_reference_logits("google/gemma-4-31B-it", ids, compute)
    assert len(calls) == 1 and torch.equal(a, b)
    files = os.listdir(tmp_path / "ref")
    assert files == [hf_reference_key("google/gemma-4-31B-it", ids)]  # no temp file left behind
    other = ids.clone()
    other[0, 3] = 999
    hf_reference_logits("google/gemma-4-31B-it", other, compute)
    assert len(calls) == 2 and len(os.listdir(tmp_path / "ref")) == 2


def test_prompt_mismatch_in_a_file_recomputes(tmp_path, monkeypatch, ids):
    monkeypatch.setenv("GEMMA4_HF_REF_CACHE", str(tmp_path))
    path = tmp_path / hf_reference_key("m", ids)
    torch.save({"input_ids": ids.clone().fill_(7), "logits": torch.zeros(1, 32, 4), "model": "m"}, path)
    out = hf_reference_logits("m", ids, lambda: torch.ones(1, 32, 4))
    assert torch.equal(out, torch.ones(1, 32, 4))
    assert torch.equal(torch.load(path)["input_ids"], ids)  # rewritten for the real prompt


def test_no_cache_dir_computes_every_time(monkeypatch, ids):
    monkeypatch.setenv("GEMMA4_HF_REF_CACHE", "")
    monkeypatch.delenv("TT_CACHE_PATH", raising=False)
    calls = []
    for _ in range(2):
        hf_reference_logits("m", ids, lambda: calls.append(1) or torch.zeros(1))
    assert len(calls) == 2


def test_unwritable_cache_dir_keeps_the_computed_reference(tmp_path, monkeypatch, ids):
    monkeypatch.setenv("GEMMA4_HF_REF_CACHE", str(tmp_path / "file_not_dir" / "x"))
    (tmp_path / "file_not_dir").write_text("not a directory")
    out = hf_reference_logits("m", ids, lambda: torch.full((1,), 3.0))
    assert float(out) == 3.0
