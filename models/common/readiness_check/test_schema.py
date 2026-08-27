# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

from models.common.readiness_check.schema import FORMAT_VERSION, load_reference


def test_load_reference_uses_weights_only(monkeypatch, tmp_path):
    payload = {
        "format_version": FORMAT_VERSION,
        "k": 2,
        "hf_model_id": "unit-test",
        "token_ids_meta": {},
        "entries": [
            {
                "prompt_text": "prompt",
                "prompt_tokens": torch.tensor([[1, 2]], dtype=torch.int64),
                "generated_tokens": torch.tensor([[3]], dtype=torch.int64),
                "topk_tokens": torch.tensor([[3, 4]], dtype=torch.int32),
                "tf_prompt_len": 2,
            }
        ],
    }

    def fake_load(path, *, weights_only):
        assert path == tmp_path / "reference.refpt"
        assert weights_only is True
        return payload

    monkeypatch.setattr(torch, "load", fake_load)

    reference = load_reference(tmp_path / "reference.refpt")
    assert reference.hf_model_id == "unit-test"
    assert reference.entries[0].generated_tokens.tolist() == [[3]]
