# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
from pathlib import Path

from models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm import Gemma4ForCausalLM
from models.common.sampling import SamplingParams


def test_capabilities_and_context_contract():
    assert Gemma4ForCausalLM.get_max_tokens_all_users() == 262_144
    assert Gemma4ForCausalLM.model_capabilities == {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_async_decode_overlap": True,
        "supports_sample_on_device": True,
        "max_device_top_k": 32,
        "state_slots_are_stateless": True,
    }


def test_adapter_delegates_sampling_and_decode():
    source = inspect.getsource(Gemma4ForCausalLM)
    assert "gen.prefill_forward" in source
    assert "gen.decode_forward" in source
    assert "gen.sample_device_logits" in source
    assert "gen._read_tokens" in source
    forbidden = ("argmax", "topk", "top_k(logits", "full_logits")
    assert all(fragment not in source.lower() for fragment in forbidden)


def test_adapter_has_no_sampling_implementation():
    tree = ast.parse(inspect.getsource(Gemma4ForCausalLM))
    methods = {node.name for node in tree.body[0].body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "sample" not in methods
    assert "sampling" not in methods


def test_selected_precision_and_external_cache_are_explicit():
    source = inspect.getsource(Gemma4ForCausalLM.initialize_vllm_model)
    assert "precision_config_path=PRECISION_CONFIG" in source
    assert "create_kv_cache=False" in source
    allocation = inspect.getsource(Gemma4ForCausalLM.allocate_kv_cache_per_layer)
    assert "gen.model.kv_cache_dtype" in allocation
    assert "FullModelState" in allocation


def test_plugin_registration_targets_autoport():
    platform = Path("../vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py").read_text()
    target = "models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm:Gemma4ForCausalLM"
    assert target in platform


def test_async_split_is_implemented():
    assert hasattr(Gemma4ForCausalLM, "read_decode_output")
    assert hasattr(Gemma4ForCausalLM, "process_decode_output_host")
    decode = inspect.getsource(Gemma4ForCausalLM.decode_forward)
    assert "read_from_device" in decode
    assert "enable_trace=True" in decode
    assert "self._page_table_refreshes" in inspect.getsource(Gemma4ForCausalLM._refresh_page_tables)


def test_unseeded_greedy_request_uses_deterministic_device_seed_default():
    params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0, seed=None)
    assert Gemma4ForCausalLM._sampling_values(params, 1)["seeds"] == (0,)


def test_unseeded_stochastic_rows_use_distinct_device_streams():
    params = SamplingParams(
        temperature=[1.0] * 4,
        top_k=[10] * 4,
        top_p=[1.0] * 4,
        seed=[None] * 4,
    )
    assert Gemma4ForCausalLM._sampling_values(params, 4)["seeds"] == (0, 1, 2, 3)
    second = Gemma4ForCausalLM._sampling_values(params, 4, unseeded_epoch=1)["seeds"]
    assert second == (104729, 104730, 104731, 104732)


def test_host_sampling_compatibility_formats_rank2_logits():
    prefill = inspect.getsource(Gemma4ForCausalLM.prefill_forward)
    decode = inspect.getsource(Gemma4ForCausalLM.decode_forward)
    assert "_host_prefill_logits(logits, len(lengths))" in prefill
    assert "_gather_logits_to_torch(output).reshape(execution_batch, 1, -1)" in decode


def test_mixed_prefill_delegates_each_sampling_output_to_generator():
    source = inspect.getsource(Gemma4ForCausalLM.prefill_forward)
    assert "if isinstance(logits, list):" in source
    assert "gen.sample_device_logits(row_logits" in source


def test_full_model_multi_user_prefill_uses_single_user_kernel_rows():
    from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import Gemma4Generator

    source = inspect.getsource(Gemma4Generator.prefill_forward)
    assert "if len(prompt_lens) > 1:" in source
    assert "user_id=row" in source


def test_chunked_prefill_propagates_absolute_positions_and_scheduler_tables():
    from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import Gemma4Generator

    adapter = inspect.getsource(Gemma4ForCausalLM.prefill_forward)
    generator = inspect.getsource(Gemma4Generator.prefill_forward)
    assert "start_pos=positions" in adapter
    assert "chunk_page_tables=self._chunk_page_tables" in adapter
    assert "end - start" in adapter
    assert "tokens[row, start:end]" in adapter
    assert "torch.arange(start, end" in adapter
    assert "return None" in inspect.getsource(Gemma4ForCausalLM._chunk_page_tables)
    assert "position_rows[row]" in generator
    assert "chunk_page_tables=chunk_page_tables" in generator
    assert "physical_len = _padded_prefill_len(logical_len)" in generator
    assert "first_position + physical_len" in generator


def test_slot_remap_is_validated_and_recaptures_without_a_second_rng_path():
    source = inspect.getsource(Gemma4ForCausalLM.decode_forward)
    assert "slot_remap must be a permutation" in source
    assert "remap_changed" in source
    assert "SeedManager" not in source
    assert "slot remapping is not yet supported" not in source


def test_padded_decode_uses_only_the_contiguous_logical_batch():
    source = inspect.getsource(Gemma4ForCausalLM.decode_forward)
    assert "logical_batch = int((flat_positions >= 0).sum().item())" in source
    assert "execution_batch = 1 if logical_batch == 1 else self.max_batch_size" in source
    assert "pack active requests before inactive slots" in source
    assert "tokens.reshape(-1)[:execution_batch].reshape(execution_batch, 1)" in source


def test_state_slot_contract_keeps_external_cache_in_execution_row_order():
    prefill = inspect.getsource(Gemma4ForCausalLM.prefill_forward)
    capabilities = Gemma4ForCausalLM.model_capabilities
    assert capabilities["state_slots_are_stateless"] is True
    assert "one unique state slot per prefill row" in prefill
    assert "state slot outside the serving batch" in prefill
    assert "external page tables are already packed" in prefill
