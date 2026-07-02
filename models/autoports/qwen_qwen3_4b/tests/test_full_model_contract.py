# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import inspect

from models.autoports.qwen_qwen3_4b.tt.generator import Qwen3Generator, build_generator
from models.autoports.qwen_qwen3_4b.tt.model import Qwen3FullModelConfig
from models.common.readiness_check.contract import Generator
from models.common.readiness_check.generate import _chat_or_plain_prompt_tokens


def test_generator_declares_readiness_contract_keywords():
    assert issubclass(Qwen3Generator, Generator)
    signature = inspect.signature(Qwen3Generator.generate)
    assert "next_input" in signature.parameters
    assert "enable_trace" in signature.parameters
    assert signature.parameters["enable_trace"].kind is inspect.Parameter.KEYWORD_ONLY


def test_build_generator_factory_is_callable():
    assert callable(build_generator)
    signature = inspect.signature(build_generator)
    assert "model_dir" in signature.parameters
    assert "mesh_device" in signature.parameters


def test_full_model_config_preserves_optimized_multichip_defaults():
    cfg = Qwen3FullModelConfig()
    assert cfg.max_seq_len == 40960
    assert cfg.paged_kv_config.cache_dtype.name == "BFLOAT16"
    assert cfg.attention_weight_dtype.name == "BFLOAT4_B"
    assert cfg.mlp_weight_dtype.name == "BFLOAT4_B"
    assert cfg.attention_math_fidelity.name == "LoFi"
    assert cfg.mlp_math_fidelity.name == "LoFi"


def test_readiness_chat_template_accepts_mapping_tokenizer_output():
    class MappingTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            assert messages == [{"role": "user", "content": "prompt"}]
            assert add_generation_prompt is True
            assert tokenize is True
            return {"input_ids": [3, 4, 5], "attention_mask": [1, 1, 1]}

    assert _chat_or_plain_prompt_tokens(MappingTokenizer(), "prompt", chat_template=True) == [3, 4, 5]
