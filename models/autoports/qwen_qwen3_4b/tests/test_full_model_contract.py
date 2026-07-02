# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import json

from models.autoports.qwen_qwen3_4b.tt.generator import Qwen3Generator, build_generator
from models.autoports.qwen_qwen3_4b.tt.model import (
    Qwen3FullModel,
    Qwen3FullModelConfig,
    load_precision_config_for_model,
)
from models.common.readiness_check.contract import Generator
from models.common.readiness_check.generate import _chat_or_plain_prompt_tokens

MODEL_DIR = "models/autoports/qwen_qwen3_4b"


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


def test_generator_exposes_low_level_traced_token_out_api():
    assert hasattr(Qwen3Generator, "prepare_token_out_decode")
    assert hasattr(Qwen3Generator, "decode_next_token_traced")
    assert hasattr(Qwen3Generator, "refresh_decode_page_table")
    assert hasattr(Qwen3Generator, "benchmark_token_out_no_readback")
    assert hasattr(Qwen3Generator, "sample_logits_topk_on_device")

    prepare_sig = inspect.signature(Qwen3Generator.prepare_token_out_decode)
    assert "page_table" in prepare_sig.parameters
    assert "read_first_token" in prepare_sig.parameters
    assert "first_input_tokens" in prepare_sig.parameters
    assert "start_positions" in prepare_sig.parameters
    assert "prompt_lens" in prepare_sig.parameters

    decode_sig = inspect.signature(Qwen3Generator.decode_next_token_traced)
    assert "page_table" in decode_sig.parameters
    assert "page_table_generation" in decode_sig.parameters


def test_token_out_prepare_normalizes_mixed_slot_state():
    tokens, positions, prompt_lens = Qwen3Generator._normalize_token_out_inputs(
        first_input_token=None,
        first_input_tokens=[11, 22],
        start_pos=None,
        start_positions=[17, 23],
        prompt_len=None,
        prompt_lens=[16, 20],
    )
    assert tuple(tokens.shape) == (2, 1)
    assert tokens.reshape(-1).tolist() == [11, 22]
    assert positions.tolist() == [17, 23]
    assert prompt_lens == [16, 20]


def test_custom_greedy_sampler_carries_active_batch_rows():
    source = inspect.getsource(Qwen3Generator)
    assert "active_batch_size" in source
    assert "_active_batch_size_from_token_tensor" in source
    assert "batch_size=self._active_batch_size_from_token_tensor(tt_out_tok)" in source

    tile_kernel = "models/autoports/qwen_qwen3_4b/tt/kernels/qwen_argmax_tile_local_winner_kernel.cpp"
    pair_kernel = "models/autoports/qwen_qwen3_4b/tt/kernels/qwen_argmax_pair_reduce_kernel.cpp"
    with open(tile_kernel, encoding="utf-8") as handle:
        tile_source = handle.read()
    with open(pair_kernel, encoding="utf-8") as handle:
        pair_source = handle.read()
    assert "active_batch_size" in tile_source
    assert "tile_offset(batch, col)" in tile_source
    assert "batch * num_senders + sender_idx" in tile_source
    assert "active_batch_size" in pair_source
    assert "output_accessor.get_noc_addr(0) + batch * sizeof(uint32_t)" in pair_source


def test_generator_keeps_distinct_greedy_and_topk_sampler_contracts():
    source = inspect.getsource(Qwen3Generator.__init__)
    assert "self.greedy_sampler" in source
    assert "max_top_k=32" in source
    assert "self.topk_sampler" in source
    assert "max_top_k=32" in source


def test_token_out_steady_loop_does_not_read_tokens_to_host():
    decode_source = inspect.getsource(Qwen3Generator.decode_next_token_on_device)
    benchmark_source = inspect.getsource(Qwen3Generator.benchmark_token_out_no_readback)
    assert "ttnn.to_torch" not in decode_source
    assert "get_device_tensors" not in decode_source
    assert "ttnn.to_torch" not in benchmark_source
    assert "get_device_tensors" not in benchmark_source


def test_model_page_table_refresh_tracks_changed_only_updates():
    source = inspect.getsource(Qwen3FullModel.refresh_trace_page_table)
    assert "page_table_host_refreshes" in source
    assert "page_table is state.page_table" in source
    assert "ttnn.copy" in source


def test_full_model_config_preserves_optimized_multichip_defaults():
    cfg = Qwen3FullModelConfig()
    assert cfg.max_seq_len == 40960
    assert cfg.paged_kv_config.cache_dtype.name == "BFLOAT16"
    assert cfg.attention_weight_dtype.name == "BFLOAT4_B"
    assert cfg.mlp_weight_dtype.name == "BFLOAT4_B"
    assert cfg.attention_math_fidelity.name == "LoFi"
    assert cfg.mlp_math_fidelity.name == "LoFi"


def test_precision_config_file_maps_all_policy_fields(tmp_path):
    path = tmp_path / "selected_precision_config.json"
    path.write_text(
        json.dumps(
            {
                "config_id": "unit_policy",
                "weight_groups": {
                    "attention": {"dtype": "bfloat8_b", "layers": "all"},
                    "mlp": {"dtype": "bfloat4_b", "layers": "all"},
                    "lm_head": {"dtype": "bfloat8_b", "layers": "all"},
                },
                "layer_exceptions": [],
                "compute_fidelities": {
                    "attention": "HiFi2",
                    "mlp": "LoFi",
                    "lm_head": "HiFi2",
                    "auxiliary": "LoFi",
                },
                "activation_dtype": "bfloat16",
                "residual_dtype": "bfloat16",
                "ccl_dtype": "bfloat8_b",
                "kv_cache_dtype": "bfloat8_b",
                "kv_cache_block_size": 32,
                "kv_cache_max_num_blocks": 1280,
                "logits_dtype": "bfloat16",
                "sampling_dtype_assumptions": {"dtype": "bfloat16"},
            }
        ),
        encoding="utf-8",
    )

    cfg = load_precision_config_for_model(MODEL_DIR, explicit_path=path, overrides={"num_layers": 1})

    assert cfg.precision_config_id == "unit_policy"
    assert cfg.num_layers == 1
    assert cfg.attention_weight_dtype.name == "BFLOAT8_B"
    assert cfg.mlp_weight_dtype.name == "BFLOAT4_B"
    assert cfg.lm_head_weight_dtype.name == "BFLOAT8_B"
    assert cfg.attention_math_fidelity.name == "HiFi2"
    assert cfg.ccl_dtype.name == "BFLOAT8_B"
    assert cfg.paged_kv_config.cache_dtype.name == "BFLOAT8_B"
    summary = cfg.precision_summary()
    assert summary["weight_groups"]["attention"]["dtype"] == "BFLOAT8_B"
    assert summary["compute_fidelities"]["attention"] == "HiFi2"
    assert summary["kv_cache_dtype"] == "BFLOAT8_B"


def test_readiness_chat_template_accepts_mapping_tokenizer_output():
    class MappingTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            assert messages == [{"role": "user", "content": "prompt"}]
            assert add_generation_prompt is True
            assert tokenize is True
            return {"input_ids": [3, 4, 5], "attention_mask": [1, 1, 1]}

    assert _chat_or_plain_prompt_tokens(MappingTokenizer(), "prompt", chat_template=True) == [3, 4, 5]
