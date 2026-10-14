# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from models.tt_dit.encoders.gemma4 import encoder_pair
from models.tt_dit.utils.ltx import LTX25_STALE_CHAT_TEMPLATE_LEN, ltx25_text_encoder_is_stale


def _packed_encoder(path, chat_template_len):
    save_file(
        {
            "hf_asset__chat_template.jinja": torch.zeros(chat_template_len, dtype=torch.uint8),
            "model.norm.weight": torch.ones(4, dtype=torch.bfloat16),
        },
        str(path),
    )
    return str(path)


@pytest.mark.parametrize("chat_template_len, stale", [(LTX25_STALE_CHAT_TEMPLATE_LEN, True), (17466, False)])
def test_ltx25_text_encoder_stale_detection(tmp_path, chat_template_len, stale):
    assert ltx25_text_encoder_is_stale(_packed_encoder(tmp_path / "te.safetensors", chat_template_len)) is stale


def test_gemma4_encoder_cache_keyed_by_file_content(tmp_path, monkeypatch):
    """Both encoder revisions share a file name, so the cache must key on the file itself."""
    calls = []
    monkeypatch.setattr(encoder_pair.cache_module, "load_model", lambda model, **kw: calls.append(kw))
    path = _packed_encoder(tmp_path / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors", 17466)
    pair = encoder_pair.Gemma4TokenizerEncoderPair.__new__(encoder_pair.Gemma4TokenizerEncoderPair)
    pair.text_encoder_path = path
    pair.gemma_encoder = object()
    pair.parallel_config = None
    pair.mesh_device = SimpleNamespace(shape=(4, 8))
    pair.config = SimpleNamespace(num_hidden_layers=48)

    pair.load_gemma_encoder()

    assert [c["subfolder"] for c in calls] == ["text_encoder"]
    assert calls[0]["sources"] == [path]
