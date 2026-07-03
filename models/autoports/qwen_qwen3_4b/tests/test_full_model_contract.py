# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import json

import torch

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
    assert "return_device_output" in prepare_sig.parameters
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


def test_generator_default_greedy_token_out_uses_traceable_tp4_sampler():
    source = inspect.getsource(Qwen3Generator.sample_logits_on_device)
    assert "_sample_greedy_tp4_to_output" in source
    greedy_source = inspect.getsource(Qwen3Generator._sample_greedy_tp4_to_output)
    assert "self.greedy_tp4_sampler.decode_forward" in greedy_source


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


def test_vllm_adapter_declares_required_capabilities_and_selected_precision():
    from models.autoports.qwen_qwen3_4b.tt.generator_vllm import Qwen3ForCausalLM

    capabilities = Qwen3ForCausalLM.model_capabilities
    assert capabilities["supports_async_decode"] is True
    assert capabilities["supports_sample_on_device"] is True
    assert capabilities["sample_on_device_policy"] == "greedy_only"
    assert capabilities["supports_prefix_caching"] is False
    assert Qwen3ForCausalLM.sample_on_device_policy == "greedy_only"

    init_source = inspect.getsource(Qwen3ForCausalLM.initialize_vllm_model)
    assert "build_generator" in init_source
    assert "optimizations is not None" in init_source


def test_vllm_adapter_has_no_host_sampling_fallbacks():
    from models.autoports.qwen_qwen3_4b.tt import generator_vllm

    source = inspect.getsource(generator_vllm.Qwen3ForCausalLM)
    assert "torch.argmax" not in source
    assert "full-logits" not in source.lower()
    assert "host_sampling" not in source
    assert "sample_logits_on_device" in source
    assert "prepare_token_out_decode" in source
    assert "decode_next_token_traced" in source


def test_vllm_adapter_allocates_and_binds_vllm_kv_cache():
    from models.autoports.qwen_qwen3_4b.tt.generator_vllm import Qwen3ForCausalLM

    source = inspect.getsource(Qwen3ForCausalLM.allocate_kv_cache)
    assert "kv_cache_shape" in source
    assert "self.generator.kv_cache = kv_cache" in source
    assert "init_paged_kv_cache" not in source


def test_vllm_adapter_requires_explicit_vllm_kv_cache():
    from models.autoports.qwen_qwen3_4b.tt.generator_vllm import Qwen3ForCausalLM

    source = inspect.getsource(Qwen3ForCausalLM)
    assert "_bound_kv_cache" not in source

    adapter = object.__new__(Qwen3ForCausalLM)
    try:
        adapter._require_kv_cache(None)
    except ValueError as exc:
        assert "vLLM-owned kv_cache" in str(exc)
    else:
        raise AssertionError("_require_kv_cache accepted a missing cache")


def test_vllm_adapter_padded_decode_uses_scratch_blocks_for_inactive_rows():
    from models.autoports.qwen_qwen3_4b.tt.generator_vllm import Qwen3ForCausalLM

    adapter = object.__new__(Qwen3ForCausalLM)
    adapter._num_kv_blocks = 128
    page_table = torch.arange(12, dtype=torch.int32).reshape(4, 3)
    active = torch.tensor([True, False, True, False])

    remapped = adapter._page_table_with_inactive_scratch(page_table, active)

    assert page_table[1].tolist() == [3, 4, 5]
    assert remapped[0].tolist() == [0, 1, 2]
    assert remapped[1].tolist() == [125, 125, 125]
    assert remapped[2].tolist() == [6, 7, 8]
    assert remapped[3].tolist() == [127, 127, 127]


def test_vllm_adapter_host_decode_remaps_inactive_rows_at_runtime():
    from models.autoports.qwen_qwen3_4b.tt.generator_vllm import Qwen3ForCausalLM

    class FakeGenerator:
        def __init__(self):
            self.calls = []

        def decode_forward(self, tokens, start_pos, **kwargs):
            self.calls.append((tokens.clone(), start_pos.clone(), kwargs))
            return torch.tensor([[1], [2]], dtype=torch.int32)

    adapter = object.__new__(Qwen3ForCausalLM)
    adapter.generator = FakeGenerator()
    adapter._num_kv_blocks = 128
    adapter._page_table_to_tt = lambda page_table: page_table

    page_table = torch.arange(8, dtype=torch.int32).reshape(4, 2)
    kv_cache = object()
    adapter.decode_forward(
        torch.tensor([[99], [77]], dtype=torch.long),
        torch.tensor([5, -1], dtype=torch.int32),
        page_table=page_table,
        kv_cache=kv_cache,
        perform_device_sampling=False,
    )

    tokens, start_pos, kwargs = adapter.generator.calls[0]
    assert tokens.reshape(-1).tolist() == [99, 0]
    assert start_pos.tolist() == [5, 0]
    assert kwargs["kv_cache"] is kv_cache
    assert kwargs["page_table"][1].tolist() == [125, 125]
    assert kwargs["use_persistent_ccl"] is False


def test_vllm_adapter_traced_decode_carries_current_position_and_page_table_generation():
    from models.autoports.qwen_qwen3_4b.tt.generator_vllm import Qwen3ForCausalLM

    source = inspect.getsource(Qwen3ForCausalLM.decode_forward)
    assert "tokens[~active] = 0" in source
    assert "start_pos[~active] = 0" in source
    assert "start_positions=start_pos.reshape(-1).to(torch.int32)" in source
    assert "prompt_lens=[int(value) for value in start_pos.reshape(-1).tolist()]" in source
    assert "page_table_generation=self._vllm_page_table_generation" in source
    assert "decode_next_token_traced" in source


def test_vllm_adapter_traced_decode_runtime_current_position_and_page_table_generation():
    from models.autoports.qwen_qwen3_4b.tt.generator_vllm import Qwen3ForCausalLM

    class TraceState:
        trace_id = None

    class FakeModel:
        trace_state = TraceState()

    class FakeGenerator:
        def __init__(self):
            self.prepare_calls = []
            self.decode_calls = []
            self.kv_cache = None

        def prepare_token_out_decode(self, **kwargs):
            self.prepare_calls.append(kwargs)
            return torch.tensor([[11]], dtype=torch.int32)

        def decode_next_token_traced(self, **kwargs):
            self.decode_calls.append(kwargs)
            return torch.tensor([[22]], dtype=torch.int32)

    adapter = object.__new__(Qwen3ForCausalLM)
    adapter.generator = FakeGenerator()
    adapter.model = FakeModel()
    adapter._decode_trace_batch_size = None
    adapter._vllm_page_table_generation = 3
    adapter._page_table_to_tt = lambda page_table: page_table

    kv_cache = object()
    page_table = torch.zeros((1, 2), dtype=torch.int32)
    adapter.decode_forward(
        torch.tensor([[123]], dtype=torch.long),
        torch.tensor([37], dtype=torch.int32),
        page_table=page_table,
        kv_cache=kv_cache,
        reset_batch=True,
        perform_device_sampling=True,
        read_from_device=False,
    )
    prepare = adapter.generator.prepare_calls[0]
    assert prepare["start_positions"].tolist() == [37]
    assert prepare["prompt_lens"] == [37]
    assert prepare["page_table"] is page_table
    assert prepare["force_argmax"] is False
    assert adapter.generator.kv_cache is kv_cache

    adapter.model.trace_state.trace_id = object()
    adapter._vllm_page_table_generation = 4
    adapter.decode_forward(
        torch.tensor([[124]], dtype=torch.long),
        torch.tensor([38], dtype=torch.int32),
        page_table=page_table,
        kv_cache=kv_cache,
        perform_device_sampling=True,
        read_from_device=False,
    )
    decode = adapter.generator.decode_calls[0]
    assert decode["page_table"] is page_table
    assert decode["page_table_generation"] == 4
    assert decode["force_argmax"] is False


def test_readiness_chat_template_accepts_mapping_tokenizer_output():
    class MappingTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            assert messages == [{"role": "user", "content": "prompt"}]
            assert add_generation_prompt is True
            assert tokenize is True
            return {"input_ids": [3, 4, 5], "attention_mask": [1, 1, 1]}

    assert _chat_or_plain_prompt_tokens(MappingTokenizer(), "prompt", chat_template=True) == [3, 4, 5]
