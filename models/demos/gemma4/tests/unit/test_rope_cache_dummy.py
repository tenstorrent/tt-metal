# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pin the contract that lets ``create_rope_caches`` use a 1-element dummy (no device).

``create_rope_caches`` (``tt/model.py``) calls HF's ``Gemma4TextRotaryEmbedding``
with a dummy ``x`` purely to carry ``dtype``/``device``: the forward reads
``x.device``/``x.dtype`` and the ``dynamic_rope_update`` decorator reads
``x.device``, never the values or the shape. It used to pass a full
``[1, max_seq_len, hidden]`` tensor, which cost 2.62 GiB (31B) / 1.88 GiB (12B)
of host RAM plus ~3 s of RNG at every model init.

If a future transformers release starts reading ``x``'s values or shape, the
uploaded cos/sin tables would silently change. This test fails there instead.
"""

from __future__ import annotations

import os

import pytest
import torch
from transformers import AutoConfig

_CONFIGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs"))

_MAX_SEQ_LEN = 512  # host-only; the invariant does not depend on the length


def _rope_tables(cfg, dummy, max_seq_len):
    """cos/sin per layer type, cast exactly as ``create_rope_caches`` does."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    rope = Gemma4TextRotaryEmbedding(cfg)
    pos_ids = torch.arange(max_seq_len).unsqueeze(0)
    tables = {}
    for layer_type in sorted(set(cfg.layer_types)):
        cos, sin = rope(dummy, pos_ids, layer_type=layer_type)
        tables[layer_type] = (cos.to(torch.bfloat16), sin.to(torch.bfloat16))
    return tables


@pytest.mark.parametrize("model_name", ["gemma-4-12B-it", "gemma-4-31B-it"])
def test_rope_tables_ignore_dummy_shape(model_name):
    hf_config = AutoConfig.from_pretrained(os.path.join(_CONFIGS_DIR, model_name))
    cfg = getattr(hf_config, "text_config", hf_config)

    full = _rope_tables(cfg, torch.randn(1, _MAX_SEQ_LEN, cfg.hidden_size), _MAX_SEQ_LEN)
    tiny = _rope_tables(cfg, torch.empty(1, dtype=torch.float32), _MAX_SEQ_LEN)

    # Non-vacuity: both models have sliding + full layer types, and each table
    # really spans max_seq_len — so the torch.equal below compares real content.
    assert sorted(full) == sorted(tiny) and len(full) == 2
    for cos, _sin in full.values():
        assert cos.shape[-2] == _MAX_SEQ_LEN and cos.shape[-1] > 0
    for layer_type in full:
        for idx, name in enumerate(("cos", "sin")):
            assert torch.equal(full[layer_type][idx], tiny[layer_type][idx]), (
                f"{model_name}/{layer_type}: {name} table changed with the dummy tensor — "
                "transformers now reads the dummy's values or shape, so create_rope_caches "
                "can no longer pass a 1-element placeholder"
            )
