# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_4b.tt.functional_decoder import HF_MODEL_ID
from models.autoports.qwen_qwen3_4b.tt.optimized_decoder import OptimizedDecoder, PagedKVConfig
from models.common.utility_functions import comp_pcc

try:
    from tracy import signpost
except ImportError:  # pragma: no cover - tracy is optional outside profiling runs.

    def signpost(header):
        return None


SMALL_SEQ_LEN = 16
NON_ALIGNED_SEQ_LEN = 17
LARGE_SEQ_LEN = 64


@pytest.fixture(scope="module")
def hf_config():
    return AutoConfig.from_pretrained(HF_MODEL_ID)


@pytest.fixture()
def mesh_device():
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), trace_region_size=16 << 20)
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)


def _tt_tensor(tensor, mesh_device):
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _synthetic_state_dict(hf_config, layer_idx=0):
    torch.manual_seed(20260702)
    hidden = hf_config.hidden_size
    q_width = hf_config.num_attention_heads * hf_config.head_dim
    kv_width = hf_config.num_key_value_heads * hf_config.head_dim
    inter = hf_config.intermediate_size
    prefix = f"model.layers.{layer_idx}"
    scale = 0.02
    return {
        f"{prefix}.self_attn.q_proj.weight": torch.randn(q_width, hidden, dtype=torch.bfloat16) * scale,
        f"{prefix}.self_attn.k_proj.weight": torch.randn(kv_width, hidden, dtype=torch.bfloat16) * scale,
        f"{prefix}.self_attn.v_proj.weight": torch.randn(kv_width, hidden, dtype=torch.bfloat16) * scale,
        f"{prefix}.self_attn.o_proj.weight": torch.randn(hidden, q_width, dtype=torch.bfloat16) * scale,
        f"{prefix}.self_attn.q_norm.weight": torch.ones(hf_config.head_dim, dtype=torch.bfloat16),
        f"{prefix}.self_attn.k_norm.weight": torch.ones(hf_config.head_dim, dtype=torch.bfloat16),
        f"{prefix}.mlp.gate_proj.weight": torch.randn(inter, hidden, dtype=torch.bfloat16) * scale,
        f"{prefix}.mlp.up_proj.weight": torch.randn(inter, hidden, dtype=torch.bfloat16) * scale,
        f"{prefix}.mlp.down_proj.weight": torch.randn(hidden, inter, dtype=torch.bfloat16) * scale,
        f"{prefix}.input_layernorm.weight": torch.ones(hidden, dtype=torch.bfloat16),
        f"{prefix}.post_attention_layernorm.weight": torch.ones(hidden, dtype=torch.bfloat16),
    }


def _causal_mask(seq_len):
    mask = torch.full((1, 1, seq_len, seq_len), torch.finfo(torch.float32).min, dtype=torch.float32)
    return torch.triu(mask, diagonal=1)


def _run_reference_layer(layer, rotary_emb, hidden_states, seq_len):
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    position_embeddings = rotary_emb(hidden_states, position_ids)
    with torch.no_grad():
        return layer(
            hidden_states,
            attention_mask=_causal_mask(seq_len),
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )


def _run_tt_prefill(decoder, hidden_states, mesh_device, *, kv_cache=None, page_table=None):
    tt_input = _tt_tensor(hidden_states.reshape(1, 1, hidden_states.shape[1], hidden_states.shape[2]), mesh_device)
    tt_output = decoder.prefill_forward(tt_input, kv_cache=kv_cache, page_table=page_table)
    return ttnn.to_torch(tt_output).reshape_as(hidden_states).to(torch.float32)


def _assert_pcc(reference, actual, threshold):
    passing, pcc = comp_pcc(reference.to(torch.float32), actual.to(torch.float32), threshold)
    print(f"PCC={pcc}")
    assert passing, f"PCC {pcc} below threshold {threshold}"
    return float(pcc)


@pytest.mark.parametrize("seq_len", [SMALL_SEQ_LEN, NON_ALIGNED_SEQ_LEN, LARGE_SEQ_LEN])
def test_optimized_synthetic_prefill_matches_hf_layer(hf_config, mesh_device, seq_len):
    layer = Qwen3DecoderLayer(hf_config, layer_idx=0).to(dtype=torch.bfloat16).eval()
    state_dict = _synthetic_state_dict(hf_config)
    layer.load_state_dict({key.removeprefix("model.layers.0."): value for key, value in state_dict.items()})
    rotary_emb = Qwen3RotaryEmbedding(hf_config)
    hidden_states = torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

    decoder = OptimizedDecoder.from_state_dict(
        state_dict, hf_config=hf_config, layer_idx=0, mesh_device=mesh_device, max_seq_len=LARGE_SEQ_LEN
    )

    reference = _run_reference_layer(layer, rotary_emb, hidden_states, seq_len)
    actual = _run_tt_prefill(decoder, hidden_states, mesh_device)
    _assert_pcc(reference, actual, 0.97)
    assert decoder.timings.prefill_ms is not None


def test_optimized_real_weight_prefill_matches_hf(hf_config, mesh_device):
    seq_len = SMALL_SEQ_LEN
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL_ID, torch_dtype=torch.bfloat16).eval()
    state_dict = model.state_dict()
    hidden_states = torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

    decoder = OptimizedDecoder.from_state_dict(
        state_dict, hf_config=hf_config, layer_idx=0, mesh_device=mesh_device, max_seq_len=seq_len
    )

    reference = _run_reference_layer(model.model.layers[0], model.model.rotary_emb, hidden_states, seq_len)
    actual = _run_tt_prefill(decoder, hidden_states, mesh_device)
    _assert_pcc(reference, actual, 0.99)


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


@pytest.mark.parametrize("prefix_len", [SMALL_SEQ_LEN, NON_ALIGNED_SEQ_LEN])
def test_optimized_paged_decode_matches_full_hf_layer(hf_config, mesh_device, prefix_len):
    total_len = prefix_len + 1
    layer = Qwen3DecoderLayer(hf_config, layer_idx=0).to(dtype=torch.bfloat16).eval()
    state_dict = _synthetic_state_dict(hf_config)
    layer.load_state_dict({key.removeprefix("model.layers.0."): value for key, value in state_dict.items()})
    rotary_emb = Qwen3RotaryEmbedding(hf_config)
    hidden_states = torch.randn(1, total_len, hf_config.hidden_size, dtype=torch.bfloat16)

    decoder = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=0,
        mesh_device=mesh_device,
        max_seq_len=LARGE_SEQ_LEN,
        paged_kv_config=PagedKVConfig(max_num_blocks=2, block_size=32),
    )

    reference = _run_reference_layer(layer, rotary_emb, hidden_states, total_len)[:, -1:, :]
    actual = _run_tt_prefill_then_decode(decoder, hidden_states, prefix_len, mesh_device)
    _assert_pcc(reference, actual, 0.97)
    assert decoder.timings.decode_ms is not None


def test_optimized_real_weight_non_aligned_paged_decode_matches_hf(hf_config, mesh_device):
    prefix_len = NON_ALIGNED_SEQ_LEN
    total_len = prefix_len + 1
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL_ID, torch_dtype=torch.bfloat16).eval()
    hidden_states = torch.randn(1, total_len, hf_config.hidden_size, dtype=torch.bfloat16)
    decoder = OptimizedDecoder.from_state_dict(
        model.state_dict(),
        hf_config=hf_config,
        layer_idx=0,
        mesh_device=mesh_device,
        max_seq_len=LARGE_SEQ_LEN,
        paged_kv_config=PagedKVConfig(max_num_blocks=2, block_size=32),
    )

    reference = _run_reference_layer(model.model.layers[0], model.model.rotary_emb, hidden_states, total_len)[:, -1:, :]
    actual = _run_tt_prefill_then_decode(decoder, hidden_states, prefix_len, mesh_device)
    _assert_pcc(reference, actual, 0.99)


def test_optimized_batched_decode_uses_disjoint_page_rows(hf_config, mesh_device):
    batch_size = 2
    prefix_len = SMALL_SEQ_LEN
    total_len = prefix_len + 1
    layer = Qwen3DecoderLayer(hf_config, layer_idx=0).to(dtype=torch.bfloat16).eval()
    state_dict = _synthetic_state_dict(hf_config)
    layer.load_state_dict({key.removeprefix("model.layers.0."): value for key, value in state_dict.items()})
    rotary_emb = Qwen3RotaryEmbedding(hf_config)
    hidden_states = torch.randn(batch_size, total_len, hf_config.hidden_size, dtype=torch.bfloat16)
    decoder = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=0,
        mesh_device=mesh_device,
        max_seq_len=LARGE_SEQ_LEN,
        paged_kv_config=PagedKVConfig(max_num_blocks=2, block_size=32),
    )
    kv_cache = decoder.init_paged_kv_cache()
    page_table = decoder.make_identity_page_table(batch_size=batch_size)
    for user_id in range(batch_size):
        prefix = hidden_states[user_id : user_id + 1, :prefix_len, :]
        tt_prefix = _tt_tensor(prefix.reshape(1, 1, prefix_len, hf_config.hidden_size), mesh_device)
        decoder.prefill_forward(tt_prefix, kv_cache=kv_cache, page_table=page_table, user_id=user_id)

    position_cos, position_sin = decoder.position_tables_for_decode(prefix_len, batch_size=batch_size)
    current_pos = decoder.make_current_pos([prefix_len] * batch_size)
    tt_decode_input = _tt_tensor(
        hidden_states[:, prefix_len : prefix_len + 1, :].reshape(1, 1, batch_size, hf_config.hidden_size),
        mesh_device,
    )
    tt_output = decoder.decode_forward(
        tt_decode_input,
        current_pos=current_pos,
        page_table=page_table,
        kv_cache=kv_cache,
        position_cos=position_cos,
        position_sin=position_sin,
    )
    actual = ttnn.to_torch(tt_output).reshape(batch_size, 1, hf_config.hidden_size).to(torch.float32)
    reference = _run_reference_layer(layer, rotary_emb, hidden_states, total_len)[:, -1:, :]
    _assert_pcc(reference, actual, 0.97)


def test_optimized_trace_replay_is_deterministic(hf_config, mesh_device):
    prefix_len = SMALL_SEQ_LEN
    total_len = prefix_len + 1
    state_dict = _synthetic_state_dict(hf_config)
    decoder = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=0,
        mesh_device=mesh_device,
        max_seq_len=LARGE_SEQ_LEN,
        paged_kv_config=PagedKVConfig(max_num_blocks=2, block_size=32),
    )
    hidden_states = torch.randn(1, total_len, hf_config.hidden_size, dtype=torch.bfloat16)
    kv_cache = decoder.init_paged_kv_cache()
    page_table = decoder.make_identity_page_table()
    tt_prefix = _tt_tensor(
        hidden_states[:, :prefix_len, :].reshape(1, 1, prefix_len, hf_config.hidden_size), mesh_device
    )
    decoder.prefill_forward(tt_prefix, kv_cache=kv_cache, page_table=page_table)

    position_cos, position_sin = decoder.position_tables_for_decode(prefix_len)
    current_pos = decoder.make_current_pos([prefix_len])
    tt_decode_input = _tt_tensor(hidden_states[:, prefix_len:, :].reshape(1, 1, 1, hf_config.hidden_size), mesh_device)
    eager_output = decoder.decode_forward(
        tt_decode_input,
        current_pos=current_pos,
        page_table=page_table,
        kv_cache=kv_cache,
        position_cos=position_cos,
        position_sin=position_sin,
    )
    trace_id, traced_output = decoder.trace_decode_once(
        tt_decode_input,
        current_pos=current_pos,
        page_table=page_table,
        kv_cache=kv_cache,
        position_cos=position_cos,
        position_sin=position_sin,
    )
    assert trace_id is not None
    eager = ttnn.to_torch(eager_output).to(torch.float32)
    traced = ttnn.to_torch(traced_output).to(torch.float32)
    _assert_pcc(eager, traced, 0.999)
    assert decoder.timings.traced_decode_ms is not None


def test_optimized_perf_signposts(hf_config, mesh_device):
    if os.environ.get("QWEN3_4B_OPT_RUN_PERF") != "1":
        pytest.skip("set QWEN3_4B_OPT_RUN_PERF=1 to run optimized decoder perf signposts")

    prefix_len = SMALL_SEQ_LEN
    total_len = prefix_len + 1
    state_dict = _synthetic_state_dict(hf_config)
    decoder = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=0,
        mesh_device=mesh_device,
        max_seq_len=LARGE_SEQ_LEN,
        paged_kv_config=PagedKVConfig(max_num_blocks=2, block_size=32),
    )
    hidden_states = torch.randn(1, total_len, hf_config.hidden_size, dtype=torch.bfloat16)

    tt_prefill = _tt_tensor(
        hidden_states[:, :prefix_len, :].reshape(1, 1, prefix_len, hf_config.hidden_size), mesh_device
    )
    decoder.prefill_forward(tt_prefill)
    signpost("PERF_PREFILL_WARMED")
    decoder.prefill_forward(tt_prefill)
    warmed_prefill_ms = decoder.timings.prefill_ms
    signpost("PERF_PREFILL_WARMED_END")

    kv_cache = decoder.init_paged_kv_cache()
    page_table = decoder.make_identity_page_table()
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

    perf_out = os.environ.get("QWEN3_4B_OPT_PERF_OUT")
    if perf_out:
        path = Path(perf_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        with path.open("a", encoding="utf-8") as f:
            if new_file:
                f.write("model,decoder,profile,layer,mode,seq_len,decode_pos,batch,traced,host_ms\n")
            f.write(
                "Qwen/Qwen3-4B,OptimizedDecoder,"
                f"{OptimizedDecoder.optimization_profile['name']},0,prefill,{prefix_len},,1,False,"
                f"{warmed_prefill_ms:.6f}\n"
            )
            f.write(
                "Qwen/Qwen3-4B,OptimizedDecoder,"
                f"{OptimizedDecoder.optimization_profile['name']},0,decode,1,{prefix_len},1,True,"
                f"{decoder.timings.traced_decode_ms:.6f}\n"
            )


def test_optimized_runtime_has_no_host_fallback():
    assert OptimizedDecoder.optimization_profile["name"] == "qwen3_4b_optimized_decoder_single_chip_v1"
    forbidden = ("torch", "from_torch", "to_torch")
    for method in (OptimizedDecoder.prefill_forward, OptimizedDecoder.decode_forward):
        source = inspect.getsource(method)
        hits = [term for term in forbidden if term in source]
        assert hits == []


def test_optimized_default_context_matches_default_paged_cache():
    signature = inspect.signature(OptimizedDecoder.from_state_dict)
    assert signature.parameters["max_seq_len"].default == PagedKVConfig().max_seq_len


def test_optimized_rejects_context_beyond_paged_cache(hf_config, expect_error):
    with expect_error(ValueError, "paged KV capacity"):
        OptimizedDecoder.from_state_dict(
            {},
            hf_config=hf_config,
            layer_idx=0,
            mesh_device=None,
            max_seq_len=PagedKVConfig().max_seq_len + 1,
            paged_kv_config=PagedKVConfig(),
        )
