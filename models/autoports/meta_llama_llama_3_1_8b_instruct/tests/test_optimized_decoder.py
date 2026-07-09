# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.optimized_decoder import OptimizedDecoder

HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_SNAPSHOT = Path(
    "/home/mvasiljevic/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/"
    "snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)
EMITTED_BATCH = 32
LAYER_IDX = 0
PAGE_BLOCK_SIZE = 32
MAX_NUM_BLOCKS = 128


@pytest.fixture(scope="module")
def mesh_device():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=16 << 20)
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)


@pytest.fixture(scope="module")
def hf_config():
    return AutoConfig.from_pretrained(HF_MODEL, local_files_only=True)


def _pcc(expected: torch.Tensor, actual: torch.Tensor) -> float:
    expected = expected.float().flatten()
    actual = actual.float().flatten()
    expected = expected - expected.mean()
    actual = actual - actual.mean()
    denom = torch.sqrt(torch.sum(expected * expected) * torch.sum(actual * actual))
    return float(torch.sum(expected * actual) / denom)


def _synthetic_state_dict(config, layer_idx=LAYER_IDX):
    g = torch.Generator().manual_seed(20260709)
    prefix = f"model.layers.{layer_idx}."

    def randn(*shape):
        return (torch.randn(*shape, generator=g) * 0.02).to(torch.bfloat16)

    return {
        prefix + "input_layernorm.weight": torch.ones(config.hidden_size, dtype=torch.bfloat16),
        prefix + "post_attention_layernorm.weight": torch.ones(config.hidden_size, dtype=torch.bfloat16),
        prefix + "self_attn.q_proj.weight": randn(config.hidden_size, config.hidden_size),
        prefix + "self_attn.k_proj.weight": randn(config.num_key_value_heads * config.head_dim, config.hidden_size),
        prefix + "self_attn.v_proj.weight": randn(config.num_key_value_heads * config.head_dim, config.hidden_size),
        prefix + "self_attn.o_proj.weight": randn(config.hidden_size, config.hidden_size),
        prefix + "mlp.gate_proj.weight": randn(config.intermediate_size, config.hidden_size),
        prefix + "mlp.up_proj.weight": randn(config.intermediate_size, config.hidden_size),
        prefix + "mlp.down_proj.weight": randn(config.hidden_size, config.intermediate_size),
    }


def _real_layer_state_dict(layer_idx=LAYER_IDX):
    index_path = MODEL_SNAPSHOT / "model.safetensors.index.json"
    with index_path.open() as f:
        weight_map = json.load(f)["weight_map"]
    keys = [
        f"model.layers.{layer_idx}.input_layernorm.weight",
        f"model.layers.{layer_idx}.post_attention_layernorm.weight",
        f"model.layers.{layer_idx}.self_attn.q_proj.weight",
        f"model.layers.{layer_idx}.self_attn.k_proj.weight",
        f"model.layers.{layer_idx}.self_attn.v_proj.weight",
        f"model.layers.{layer_idx}.self_attn.o_proj.weight",
        f"model.layers.{layer_idx}.mlp.gate_proj.weight",
        f"model.layers.{layer_idx}.mlp.up_proj.weight",
        f"model.layers.{layer_idx}.mlp.down_proj.weight",
    ]
    loaded = {}
    for shard in sorted({weight_map[key] for key in keys}):
        loaded.update(load_file(str(MODEL_SNAPSHOT / shard)))
    return {key: loaded[key].to(torch.bfloat16) for key in keys}


def _reference_layer(config, state_dict, layer_idx=LAYER_IDX):
    layer = LlamaDecoderLayer(config, layer_idx=layer_idx).to(dtype=torch.bfloat16)
    local = {key.removeprefix(f"model.layers.{layer_idx}."): value for key, value in state_dict.items()}
    layer.load_state_dict(local, strict=True)
    layer.eval()
    return layer


def _rope(config, hidden_states, seq_len):
    rotary = LlamaRotaryEmbedding(config).to(dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(hidden_states.shape[0], -1)
    return rotary(hidden_states, position_ids)


def _page_table(batch=EMITTED_BATCH, max_num_blocks=MAX_NUM_BLOCKS, *, permuted=False):
    blocks_per_user = max_num_blocks // batch
    blocks = torch.arange(max_num_blocks, dtype=torch.int32)
    if permuted:
        blocks = blocks[torch.randperm(max_num_blocks, generator=torch.Generator().manual_seed(20260709))]
    rows = blocks.reshape(batch, blocks_per_user)
    return rows


def _reference_kv_prefix(config, state_dict, hidden, cos, sin, prefix_len, layer_idx=LAYER_IDX, page_table=None):
    prefix = f"model.layers.{layer_idx}."
    ln_w = state_dict[prefix + "input_layernorm.weight"].float()
    q_w = state_dict[prefix + "self_attn.q_proj.weight"].float()
    k_w = state_dict[prefix + "self_attn.k_proj.weight"].float()
    v_w = state_dict[prefix + "self_attn.v_proj.weight"].float()
    eps = config.rms_norm_eps
    x = hidden[:, :prefix_len].float()
    normed = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * ln_w
    k = torch.matmul(normed, k_w.T)
    v = torch.matmul(normed, v_w.T)
    k = k.view(EMITTED_BATCH, prefix_len, config.num_key_value_heads, config.head_dim).transpose(1, 2)
    v = v.view(EMITTED_BATCH, prefix_len, config.num_key_value_heads, config.head_dim).transpose(1, 2)

    cos_prefix = cos[:, :prefix_len].float().unsqueeze(1)
    sin_prefix = sin[:, :prefix_len].float().unsqueeze(1)
    half = config.head_dim // 2
    k1 = k[..., :half]
    k2 = k[..., half:]
    k_rot = torch.cat((-k2, k1), dim=-1)
    k = k * cos_prefix + k_rot * sin_prefix

    cache_shape = (MAX_NUM_BLOCKS, config.num_key_value_heads, PAGE_BLOCK_SIZE, config.head_dim)
    k_cache = torch.zeros(cache_shape, dtype=torch.bfloat16)
    v_cache = torch.zeros(cache_shape, dtype=torch.bfloat16)
    if page_table is None:
        page_table = _page_table()
    for batch_idx in range(EMITTED_BATCH):
        for pos in range(prefix_len):
            block = int(page_table[batch_idx, pos // PAGE_BLOCK_SIZE])
            offset = pos % PAGE_BLOCK_SIZE
            k_cache[block, :, offset, :] = k[batch_idx, :, pos, :].to(torch.bfloat16)
            v_cache[block, :, offset, :] = v[batch_idx, :, pos, :].to(torch.bfloat16)
    return k_cache, v_cache, page_table


def _run_decode_case(hf_config, mesh_device, *, prefix_len, decode_positions, permuted_page_table=False):
    total_len = max(decode_positions) + 1
    torch.manual_seed(4321 + total_len + prefix_len)
    hidden = torch.randn(EMITTED_BATCH, total_len, hf_config.hidden_size, dtype=torch.bfloat16)
    cos, sin = _rope(hf_config, hidden, total_len)
    state_dict = _real_layer_state_dict(0)
    ref = _reference_layer(hf_config, state_dict, layer_idx=0)
    attention_mask = torch.full((EMITTED_BATCH, 1, total_len, total_len), torch.finfo(torch.float32).min)
    attention_mask = torch.triu(attention_mask, diagonal=1).to(torch.bfloat16)
    with torch.no_grad():
        expected = ref(hidden, attention_mask=attention_mask, position_embeddings=(cos, sin))

    decoder = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=0,
        mesh_device=mesh_device,
        batch=EMITTED_BATCH,
        page_block_size=PAGE_BLOCK_SIZE,
        max_num_blocks=MAX_NUM_BLOCKS,
    )
    page_table = _page_table(permuted=permuted_page_table)
    k_cache_torch, v_cache_torch, page_table = _reference_kv_prefix(
        hf_config, state_dict, hidden, cos, sin, prefix_len, layer_idx=0, page_table=page_table
    )
    k_cache = ttnn.from_torch(k_cache_torch, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    v_cache = ttnn.from_torch(v_cache_torch, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    tt_page_table = decoder.prepare_page_table(page_table)

    actuals = []
    pccs = []
    for pos in decode_positions:
        tt_hidden = OptimizedDecoder.prepare_decode_inputs(hidden[:, pos : pos + 1, :], mesh_device)
        tt_pos = decoder.prepare_current_pos(torch.full((EMITTED_BATCH,), pos, dtype=torch.int32))
        tt_cos, tt_sin = decoder.prepare_decode_rope(cos[:, pos : pos + 1, :], sin[:, pos : pos + 1, :])
        actual = ttnn.to_torch(
            decoder.decode_forward(
                tt_hidden,
                current_pos=tt_pos,
                position_cos=tt_cos,
                position_sin=tt_sin,
                kv_cache=(k_cache, v_cache),
                page_table=tt_page_table,
            )
        ).reshape(EMITTED_BATCH, 1, hf_config.hidden_size)
        pccs.append(_pcc(expected[:, pos : pos + 1, :], actual))
        actuals.append(actual)
    return pccs, actuals


def _build_prefill_runtime(hf_config, mesh_device, *, seq_len=5):
    torch.manual_seed(1234 + seq_len)
    hidden = torch.randn(EMITTED_BATCH, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)
    cos, sin = _rope(hf_config, hidden, seq_len)
    decoder = OptimizedDecoder.from_state_dict(
        _real_layer_state_dict(0),
        hf_config=hf_config,
        layer_idx=0,
        mesh_device=mesh_device,
        batch=EMITTED_BATCH,
        page_block_size=PAGE_BLOCK_SIZE,
        max_num_blocks=MAX_NUM_BLOCKS,
    )
    return (
        decoder,
        OptimizedDecoder.prepare_inputs(hidden, mesh_device),
        OptimizedDecoder.prepare_rope(cos, sin, mesh_device),
    )


def _build_decode_runtime(hf_config, mesh_device, *, prefix_len=31):
    total_len = prefix_len + 1
    torch.manual_seed(4321 + total_len + prefix_len)
    hidden = torch.randn(EMITTED_BATCH, total_len, hf_config.hidden_size, dtype=torch.bfloat16)
    cos, sin = _rope(hf_config, hidden, total_len)
    state_dict = _real_layer_state_dict(0)
    decoder = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=0,
        mesh_device=mesh_device,
        batch=EMITTED_BATCH,
        page_block_size=PAGE_BLOCK_SIZE,
        max_num_blocks=MAX_NUM_BLOCKS,
    )
    page_table = _page_table(permuted=True)
    k_cache_torch, v_cache_torch, page_table = _reference_kv_prefix(
        hf_config, state_dict, hidden, cos, sin, prefix_len, layer_idx=0, page_table=page_table
    )
    return (
        decoder,
        OptimizedDecoder.prepare_decode_inputs(hidden[:, prefix_len : prefix_len + 1, :], mesh_device),
        decoder.prepare_current_pos(torch.full((EMITTED_BATCH,), prefix_len, dtype=torch.int32)),
        decoder.prepare_decode_rope(cos[:, prefix_len : prefix_len + 1, :], sin[:, prefix_len : prefix_len + 1, :]),
        (
            ttnn.from_torch(k_cache_torch, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device),
            ttnn.from_torch(v_cache_torch, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device),
        ),
        decoder.prepare_page_table(page_table),
    )


def _signpost(label: str):
    try:
        from tracy import signpost
    except ImportError:
        return
    signpost(label)


def test_optimized_prefill_real_weight_non_aligned_pcc(hf_config, mesh_device):
    seq_len = 5
    torch.manual_seed(1234 + seq_len)
    hidden = torch.randn(EMITTED_BATCH, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)
    cos, sin = _rope(hf_config, hidden, seq_len)
    state_dict = _real_layer_state_dict(0)
    ref = _reference_layer(hf_config, state_dict, layer_idx=0)
    attention_mask = torch.full((EMITTED_BATCH, 1, seq_len, seq_len), torch.finfo(torch.float32).min)
    attention_mask = torch.triu(attention_mask, diagonal=1).to(torch.bfloat16)
    with torch.no_grad():
        expected = ref(hidden, attention_mask=attention_mask, position_embeddings=(cos, sin))

    decoder = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=0,
        mesh_device=mesh_device,
        batch=EMITTED_BATCH,
        page_block_size=PAGE_BLOCK_SIZE,
        max_num_blocks=MAX_NUM_BLOCKS,
    )
    tt_hidden = OptimizedDecoder.prepare_inputs(hidden, mesh_device)
    tt_cos, tt_sin = OptimizedDecoder.prepare_rope(cos, sin, mesh_device)
    actual = ttnn.to_torch(decoder.prefill_forward(tt_hidden, position_cos=tt_cos, position_sin=tt_sin)).squeeze(0)
    measured = _pcc(expected, actual)
    print(f"OPT_REAL_WEIGHT_PREFILL_SEQ_{seq_len}_PCC={measured:.6f}")
    assert measured >= 0.99


def test_optimized_decode_paged_real_weight_non_aligned_pcc(hf_config, mesh_device):
    prefix_len = 5
    pccs, _ = _run_decode_case(hf_config, mesh_device, prefix_len=prefix_len, decode_positions=[prefix_len])
    measured = pccs[0]
    print(f"OPT_REAL_WEIGHT_DECODE_PREFIX_{prefix_len}_PCC={measured:.6f}")
    assert measured >= 0.99


def test_optimized_decode_repeated_paged_block_boundary_deterministic(hf_config, mesh_device):
    decode_positions = [31, 32, 33]
    pccs_a, actuals_a = _run_decode_case(
        hf_config,
        mesh_device,
        prefix_len=31,
        decode_positions=decode_positions,
        permuted_page_table=True,
    )
    pccs_b, actuals_b = _run_decode_case(
        hf_config,
        mesh_device,
        prefix_len=31,
        decode_positions=decode_positions,
        permuted_page_table=True,
    )
    for pos, pcc in zip(decode_positions, pccs_a):
        print(f"OPT_REAL_WEIGHT_REPEATED_DECODE_POS_{pos}_PCC={pcc:.6f}")
        assert pcc >= 0.99
    for lhs, rhs in zip(actuals_a, actuals_b):
        assert _pcc(lhs, rhs) >= 0.9999


def test_optimized_path_has_no_functional_fallbacks():
    source = inspect.getsource(OptimizedDecoder)
    forbidden_runtime = ("FunctionalDecoder(", "FunctionalDecoder.", "functional_decoder.FunctionalDecoder")
    assert not any(token in source for token in forbidden_runtime)
    assert "paged_scaled_dot_product_attention_decode" in source
    assert "paged_update_cache" in source


def test_optimized_runtime_has_no_host_fallbacks():
    for method in (OptimizedDecoder.prefill_forward, OptimizedDecoder.decode_forward):
        source = inspect.getsource(method)
        forbidden = ("torch", "from_torch", "to_torch")
        assert not any(token in source for token in forbidden)


def test_optimized_decode_trace_replay_correctness(hf_config, mesh_device):
    decoder, tt_hidden, tt_pos, (tt_cos, tt_sin), kv_cache, tt_page_table = _build_decode_runtime(
        hf_config, mesh_device, prefix_len=31
    )
    decoder.decode_forward(
        tt_hidden,
        current_pos=tt_pos,
        position_cos=tt_cos,
        position_sin=tt_sin,
        kv_cache=kv_cache,
        page_table=tt_page_table,
    )

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        traced_out = decoder.decode_forward(
            tt_hidden,
            current_pos=tt_pos,
            position_cos=tt_cos,
            position_sin=tt_sin,
            kv_cache=kv_cache,
            page_table=tt_page_table,
        )
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        captured = ttnn.to_torch(traced_out).reshape(EMITTED_BATCH, 1, hf_config.hidden_size)
        for _ in range(3):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            replayed = ttnn.to_torch(traced_out).reshape(EMITTED_BATCH, 1, hf_config.hidden_size)
            assert _pcc(captured, replayed) >= 0.9999
    finally:
        ttnn.release_trace(mesh_device, trace_id)


def test_perf_prefill_signposted(hf_config, mesh_device):
    decoder, tt_hidden, (tt_cos, tt_sin) = _build_prefill_runtime(hf_config, mesh_device, seq_len=5)
    decoder.prefill_forward(tt_hidden, position_cos=tt_cos, position_sin=tt_sin)
    _signpost("PERF_PREFILL")
    start = time.perf_counter()
    out = decoder.prefill_forward(tt_hidden, position_cos=tt_cos, position_sin=tt_sin)
    ttnn.synchronize_device(mesh_device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    _signpost("PERF_PREFILL_END")
    print(f"OPT_PREFILL_SEQ_5_WARMED_MS={elapsed_ms:.3f}")
    assert out is not None


def test_perf_decode_signposted(hf_config, mesh_device):
    decoder, tt_hidden, tt_pos, (tt_cos, tt_sin), kv_cache, tt_page_table = _build_decode_runtime(
        hf_config, mesh_device, prefix_len=31
    )
    decoder.decode_forward(
        tt_hidden,
        current_pos=tt_pos,
        position_cos=tt_cos,
        position_sin=tt_sin,
        kv_cache=kv_cache,
        page_table=tt_page_table,
    )
    _signpost("PERF_DECODE")
    start = time.perf_counter()
    out = decoder.decode_forward(
        tt_hidden,
        current_pos=tt_pos,
        position_cos=tt_cos,
        position_sin=tt_sin,
        kv_cache=kv_cache,
        page_table=tt_page_table,
    )
    ttnn.synchronize_device(mesh_device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    _signpost("PERF_DECODE_END")
    print(f"OPT_DECODE_PREFIX_31_WARMED_MS={elapsed_ms:.3f}")
    assert out is not None


def test_perf_decode_trace_signposted(hf_config, mesh_device):
    decoder, tt_hidden, tt_pos, (tt_cos, tt_sin), kv_cache, tt_page_table = _build_decode_runtime(
        hf_config, mesh_device, prefix_len=31
    )
    decoder.decode_forward(
        tt_hidden,
        current_pos=tt_pos,
        position_cos=tt_cos,
        position_sin=tt_sin,
        kv_cache=kv_cache,
        page_table=tt_page_table,
    )

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        out = decoder.decode_forward(
            tt_hidden,
            current_pos=tt_pos,
            position_cos=tt_cos,
            position_sin=tt_sin,
            kv_cache=kv_cache,
            page_table=tt_page_table,
        )
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        _signpost("PERF_DECODE_TRACE")
        start = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _signpost("PERF_DECODE_TRACE_END")
        print(f"OPT_DECODE_TRACE_PREFIX_31_WARMED_MS={elapsed_ms:.3f}")
        assert out is not None
    finally:
        ttnn.release_trace(mesh_device, trace_id)
