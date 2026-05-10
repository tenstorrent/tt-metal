# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.experimental.mistral_small_4_119b.constants import HF_MODEL_ID
import models.experimental.mistral_small_4_119b as ms4


pytest.importorskip("transformers")


def test_hf_text_config_matches_architecture_contract():
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Could not download HF config: {exc}")

    text = getattr(cfg, "text_config", None)
    assert text is not None, "Mistral3-style config should expose text_config"

    assert text.hidden_size == ms4.EXPECTED_HIDDEN_SIZE
    assert text.num_hidden_layers == ms4.EXPECTED_NUM_LAYERS
    assert text.vocab_size == ms4.EXPECTED_VOCAB_SIZE

    for attr in ("num_local_experts", "num_experts"):
        if hasattr(text, attr):
            assert getattr(text, attr) == ms4.EXPECTED_NUM_EXPERTS
            break
    else:
        pytest.fail("text_config has no num_local_experts / num_experts")

    for attr in ("num_experts_per_tok", "moe_topk", "num_expert_per_tok"):
        if hasattr(text, attr):
            assert getattr(text, attr) == ms4.EXPECTED_NUM_EXPERTS_PER_TOK
            break
    else:
        pytest.fail("text_config has no experts-per-token field")
