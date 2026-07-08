# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tests.test_optimized_decoder import (
    LARGE_SEQ_LEN,
    NON_ALIGNED_SEQ_LEN,
    SMALL_SEQ_LEN,
    _assert_pcc,
    _build_synthetic_layer,
    _load_real_model_or_skip,
    _run_reference_layer,
    _synthetic_state_dict,
    _tt_tensor,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.optimized_decoder import PagedKVConfig
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.optimized_decoder_force_emitted_configs import (
    OptimizedDecoderForceEmittedConfigs,
)


def _build_decoder(state_dict, hf_config, mesh_device, *, layer_idx=0, force_scope="unchanged"):
    decoder = OptimizedDecoderForceEmittedConfigs.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_seq_len=LARGE_SEQ_LEN,
        paged_kv_config=PagedKVConfig(max_num_blocks=2, block_size=32),
    )
    decoder.force_scope = force_scope
    return decoder


def _run_tt_prefill_then_decode(decoder, hidden_states, prefix_len, mesh_device):
    kv_cache = decoder.init_paged_kv_cache()
    page_table = decoder.make_identity_page_table()
    prefix = hidden_states[:, :prefix_len, :]
    decode_input = hidden_states[:, prefix_len : prefix_len + 1, :]

    tt_prefix = _tt_tensor(prefix.reshape(1, 1, prefix_len, hidden_states.shape[-1]), mesh_device)
    decoder.prefill_forward(tt_prefix, kv_cache=kv_cache, page_table=page_table)

    position_cos, position_sin = decoder.position_tables_for_decode(prefix_len)
    current_pos = decoder.make_current_pos([prefix_len])
    tt_decode_input = _tt_tensor(decode_input.reshape(1, 1, 1, hidden_states.shape[-1]), mesh_device)
    tt_output = decoder.decode_forward(
        tt_decode_input,
        current_pos=current_pos,
        page_table=page_table,
        kv_cache=kv_cache,
        position_cos=position_cos,
        position_sin=position_sin,
    )
    return ttnn.to_torch(tt_output).reshape_as(decode_input).to(torch.float32)


@pytest.mark.parametrize("force_scope", ["unchanged", "all"])
def test_force_emitted_synthetic_paged_decode_matches_hf(hf_config, mesh_device, force_scope):
    prefix_len = NON_ALIGNED_SEQ_LEN
    total_len = prefix_len + 1
    layer, state_dict = _build_synthetic_layer(hf_config)
    rotary_emb = LlamaRotaryEmbedding(hf_config)
    hidden_states = torch.randn(1, total_len, hf_config.hidden_size, dtype=torch.bfloat16)

    decoder = _build_decoder(state_dict, hf_config, mesh_device, force_scope=force_scope)
    reference = _run_reference_layer(layer, rotary_emb, hidden_states, total_len)[:, -1:, :]
    actual = _run_tt_prefill_then_decode(decoder, hidden_states, prefix_len, mesh_device)
    _assert_pcc(reference, actual, 0.95)
    assert decoder.timings.decode_ms is not None


@pytest.mark.parametrize("force_scope", ["unchanged", "all"])
def test_force_emitted_real_weight_paged_decode_matches_hf(mesh_device, force_scope):
    prefix_len = NON_ALIGNED_SEQ_LEN
    total_len = prefix_len + 1
    model = _load_real_model_or_skip()
    hf_config = model.config
    hidden_states = torch.randn(1, total_len, hf_config.hidden_size, dtype=torch.bfloat16)
    decoder = _build_decoder(model.state_dict(), hf_config, mesh_device, force_scope=force_scope)

    reference = _run_reference_layer(model.model.layers[0], model.model.rotary_emb, hidden_states, total_len)[:, -1:, :]
    actual = _run_tt_prefill_then_decode(decoder, hidden_states, prefix_len, mesh_device)
    _assert_pcc(reference, actual, 0.99)


def test_force_emitted_perf_signposts(hf_config, mesh_device):
    if os.environ.get("LLAMA31_8B_OPT_RUN_PERF") != "1":
        pytest.skip("set LLAMA31_8B_OPT_RUN_PERF=1 to run force-emitted decoder perf signposts")

    force_scope = os.environ.get("LLAMA31_8B_FORCE_SCOPE", "unchanged")
    prefix_len = SMALL_SEQ_LEN
    state_dict = _synthetic_state_dict(hf_config)
    decoder = _build_decoder(state_dict, hf_config, mesh_device, force_scope=force_scope)
    hidden_states = torch.randn(1, prefix_len + 1, hf_config.hidden_size, dtype=torch.bfloat16)

    kv_cache = decoder.init_paged_kv_cache()
    page_table = decoder.make_identity_page_table()
    tt_prefill = _tt_tensor(
        hidden_states[:, :prefix_len, :].reshape(1, 1, prefix_len, hf_config.hidden_size), mesh_device
    )
    decoder.prefill_forward(tt_prefill, kv_cache=kv_cache, page_table=page_table)
    position_cos, position_sin = decoder.position_tables_for_decode(prefix_len)
    current_pos = decoder.make_current_pos([prefix_len])
    tt_decode_input = _tt_tensor(hidden_states[:, prefix_len:, :].reshape(1, 1, 1, hf_config.hidden_size), mesh_device)
    trace_id, _ = decoder.trace_decode_once(
        tt_decode_input,
        current_pos=current_pos,
        page_table=page_table,
        kv_cache=kv_cache,
        position_cos=position_cos,
        position_sin=position_sin,
    )
    assert trace_id is not None
    print(f"FORCE_EMITTED[{force_scope}] traced_decode_ms={decoder.timings.traced_decode_ms}")

    perf_out = os.environ.get("LLAMA31_8B_OPT_PERF_OUT")
    if perf_out:
        path = Path(perf_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        with path.open("a", encoding="utf-8") as f:
            if new_file:
                f.write("model,decoder,profile,force_scope,layer,mode,seq_len,decode_pos,batch,traced,host_ms\n")
            f.write(
                "meta-llama/Llama-3.1-8B-Instruct,OptimizedDecoderForceEmittedConfigs,"
                f"{OptimizedDecoderForceEmittedConfigs.optimization_profile['name']},{force_scope},0,decode,1,"
                f"{prefix_len},1,True,{decoder.timings.traced_decode_ms:.6f}\n"
            )
