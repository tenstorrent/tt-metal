# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import json
import math
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer, Gemma4TextRotaryEmbedding

import ttnn
from models.autoports.google_gemma_4_31b.tt.functional_decoder import HF_MODEL_ID, FunctionalDecoder
from models.common.utility_functions import comp_pcc
from models.demos.gemma4.tt.attention import Gemma4Attention, Gemma4AttentionConfig
from models.demos.gemma4.tt.attention.decode import decode_forward as gemma_decode_forward
from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
from models.demos.gemma4.tt.rms_norm import RMSNorm
from models.demos.gemma4.tt.shared_mlp import SharedMLP
from models.tt_transformers.tt.common import PagedAttentionConfig

LAYER_KINDS = ((0, "sliding_attention"), (5, "full_attention"))
BLOCK_SIZE = 64


def _checkpoint_dir() -> Path:
    hub = Path.home() / ".cache/huggingface/hub/models--google--gemma-4-31B/snapshots"
    snapshots = sorted(hub.glob("*"))
    if not snapshots:
        pytest.skip(f"real {HF_MODEL_ID} checkpoint is not cached")
    return snapshots[-1]


@pytest.fixture(scope="module")
def hf_config():
    return AutoConfig.from_pretrained(_checkpoint_dir(), trust_remote_code=True, local_files_only=True).text_config


@pytest.fixture(scope="module")
def mesh_device():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)


def _layer_state(layer_idx: int) -> dict[str, torch.Tensor]:
    root = _checkpoint_dir()
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer_idx}."
    result = {}
    by_shard = {}
    for key, shard in index.items():
        if key.startswith(prefix):
            by_shard.setdefault(shard, []).append(key)
    for shard, keys in by_shard.items():
        with safe_open(root / shard, framework="pt", device="cpu") as handle:
            for key in keys:
                result[key] = handle.get_tensor(key)
    if not result:
        raise RuntimeError(f"checkpoint contains no weights for layer {layer_idx}")
    return result


def _local_state(full_state, layer_idx):
    prefix = f"model.language_model.layers.{layer_idx}."
    return {key.removeprefix(prefix): value for key, value in full_state.items()}


def _rope_host(hf_config, layer_idx, positions):
    rope = Gemma4TextRotaryEmbedding(hf_config)
    pos = torch.as_tensor(positions, dtype=torch.long).reshape(1, -1)
    dummy = torch.zeros(1, pos.shape[1], 1, dtype=torch.bfloat16)
    return rope(dummy, pos, layer_type=hf_config.layer_types[layer_idx])


def _rope_device(hf_config, layer_idx, max_seq_len, mesh_device, *, decode):
    cos, sin = _rope_host(hf_config, layer_idx, torch.arange(max_seq_len))
    if decode:
        cos, sin = cos.squeeze(0), sin.squeeze(0)
        layout = ttnn.ROW_MAJOR_LAYOUT
    else:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
        layout = ttnn.TILE_LAYOUT
    return tuple(ttnn.from_torch(t, device=mesh_device, dtype=ttnn.bfloat16, layout=layout) for t in (cos, sin))


def _paged_state(hf_config, layer_idx, max_context, mesh_device, *, permutation=True, batch_size=1):
    args = Gemma4ModelArgs.from_hf_config(hf_config)
    cfg = Gemma4AttentionConfig(args, layer_idx)
    physical_context = cfg.sliding_window if cfg.is_sliding else max_context
    num_blocks = math.ceil(physical_context / BLOCK_SIZE)
    paged = PagedAttentionConfig(block_size=BLOCK_SIZE, max_num_blocks=num_blocks * batch_size)
    cache = init_kv_cache(
        mesh_device,
        cfg,
        paged_attention_config=paged,
        cache_dtype=ttnn.bfloat16,
        max_num_blocks_override=num_blocks * batch_size,
    )
    rows = []
    for batch_idx in range(batch_size):
        block_ids = torch.arange(batch_idx * num_blocks, (batch_idx + 1) * num_blocks, dtype=torch.int32)
        if permutation:
            block_ids = torch.roll(block_ids, shifts=1)
        rows.append(block_ids)
    page_table = ttnn.from_torch(torch.stack(rows), device=mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    return cache, page_table


def _tt_input(x, mesh_device):
    return ttnn.from_torch(
        x.reshape(1, 1, x.shape[1], x.shape[2]),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _to_host(x):
    return ttnn.to_torch(x).reshape(1, x.shape[-2], x.shape[-1]).float()


def _assert_pcc(reference, actual, threshold=0.995):
    passing, pcc = comp_pcc(reference.float(), actual.float(), threshold)
    print(f"PCC={pcc}")
    assert passing, f"PCC {pcc} below {threshold}"
    return float(pcc)


def _hf_prefill(layer, hf_config, layer_idx, hidden):
    positions = torch.arange(hidden.shape[1]).reshape(1, -1)
    rope = _rope_host(hf_config, layer_idx, positions)
    causal = torch.triu(
        torch.full((1, 1, hidden.shape[1], hidden.shape[1]), float("-inf"), dtype=torch.float32), diagonal=1
    )
    if hf_config.layer_types[layer_idx] == "sliding_attention" and hidden.shape[1] > hf_config.sliding_window:
        idx = torch.arange(hidden.shape[1])
        outside = idx.unsqueeze(0) < idx.unsqueeze(1) - hf_config.sliding_window + 1
        causal.masked_fill_(outside.unsqueeze(0).unsqueeze(0), float("-inf"))
    with torch.no_grad():
        return layer(hidden, position_embeddings=rope, attention_mask=causal)


def _hf_prefill_absolute(layer, hf_config, layer_idx, hidden, start_position):
    positions = torch.arange(start_position, start_position + hidden.shape[1]).reshape(1, -1)
    rope = _rope_host(hf_config, layer_idx, positions)
    causal = torch.triu(
        torch.full((1, 1, hidden.shape[1], hidden.shape[1]), float("-inf"), dtype=torch.float32), diagonal=1
    )
    with torch.no_grad():
        return layer(hidden, position_embeddings=rope, attention_mask=causal)


@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", [32, 33])
def test_real_weight_paged_prefill_pcc(hf_config, mesh_device, layer_idx, layer_kind, seq_len):
    state = _layer_state(layer_idx)
    hf_layer = Gemma4TextDecoderLayer(hf_config, layer_idx).to(torch.bfloat16).eval()
    hf_layer.load_state_dict(_local_state(state, layer_idx))
    decoder = FunctionalDecoder.from_state_dict(
        state, hf_config=hf_config, layer_idx=layer_idx, mesh_device=mesh_device
    )
    cache, page_table = _paged_state(hf_config, layer_idx, max(seq_len, 128), mesh_device)
    rope = _rope_device(hf_config, layer_idx, seq_len, mesh_device, decode=False)
    torch.manual_seed(20260711 + layer_idx + seq_len)
    hidden = torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
    reference = _hf_prefill(hf_layer, hf_config, layer_idx, hidden)
    actual_tt = decoder.prefill_forward(
        _tt_input(hidden, mesh_device), rope_mats=rope, page_table=page_table, kv_cache=cache, valid_seq_len=seq_len
    )
    _assert_pcc(reference, _to_host(actual_tt))


@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
def test_real_weight_paged_decode_trace_pcc(hf_config, mesh_device, layer_idx, layer_kind):
    prompt_len = 32
    state = _layer_state(layer_idx)
    hf_layer = Gemma4TextDecoderLayer(hf_config, layer_idx).to(torch.bfloat16).eval()
    hf_layer.load_state_dict(_local_state(state, layer_idx))
    decoder = FunctionalDecoder.from_state_dict(
        state, hf_config=hf_config, layer_idx=layer_idx, mesh_device=mesh_device
    )
    cache, page_table = _paged_state(hf_config, layer_idx, 128, mesh_device)
    prefill_rope = _rope_device(hf_config, layer_idx, prompt_len, mesh_device, decode=False)
    decode_rope = _rope_device(hf_config, layer_idx, 128, mesh_device, decode=True)
    torch.manual_seed(20260721 + layer_idx)
    prompt = torch.randn(1, prompt_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
    token = torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
    decoder.prefill_forward(
        _tt_input(prompt, mesh_device),
        rope_mats=prefill_rope,
        page_table=page_table,
        kv_cache=cache,
        valid_seq_len=prompt_len,
    )
    pos_u_host = torch.zeros((1, 32), dtype=torch.int32)
    pos_u_host[0, 0] = prompt_len
    pos_u = ttnn.from_torch(pos_u_host, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    pos_i = ttnn.from_torch(
        torch.tensor([prompt_len], dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    tt_token = _tt_input(token, mesh_device)

    dynamic_cache = DynamicCache()
    _hf_prefill(hf_layer, hf_config, layer_idx, prompt)
    prefill_positions = torch.arange(prompt_len).reshape(1, -1)
    prefill_rope_hf = _rope_host(hf_config, layer_idx, prefill_positions)
    causal = torch.triu(torch.full((1, 1, prompt_len, prompt_len), float("-inf")), diagonal=1)
    with torch.no_grad():
        hf_layer(
            prompt,
            position_embeddings=prefill_rope_hf,
            attention_mask=causal,
            past_key_values=dynamic_cache,
        )
        decode_rope_hf = _rope_host(hf_config, layer_idx, [prompt_len])
        reference = hf_layer(
            token,
            position_embeddings=decode_rope_hf,
            attention_mask=torch.zeros(1, 1, 1, prompt_len + 1),
            past_key_values=dynamic_cache,
        )

    def decode_call():
        return decoder.decode_forward(
            tt_token,
            rope_mats=decode_rope,
            page_table=page_table,
            kv_cache=cache,
            current_position=pos_u,
            current_position_cache=pos_i,
            token_index=0,
        )

    warm = decode_call()
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decode_call()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        replay_one = _to_host(traced_output).clone()
        _assert_pcc(reference, replay_one)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        replay_two = _to_host(traced_output).clone()
        assert torch.equal(replay_one, replay_two), "identical traced decode replay was not deterministic"
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
def test_real_shape_prefill_boundaries(hf_config, mesh_device, layer_idx, layer_kind):
    state = _layer_state(layer_idx)
    decoder = FunctionalDecoder.from_state_dict(
        state, hf_config=hf_config, layer_idx=layer_idx, mesh_device=mesh_device
    )
    hf_layer = Gemma4TextDecoderLayer(hf_config, layer_idx).to(torch.bfloat16).eval()
    hf_layer.load_state_dict(_local_state(state, layer_idx))
    for seq_len in (63, 64, 65, 1023, 1024, 1025):
        cache, page_table = _paged_state(hf_config, layer_idx, seq_len, mesh_device)
        rope = _rope_device(hf_config, layer_idx, seq_len, mesh_device, decode=False)
        torch.manual_seed(20260800 + layer_idx + seq_len)
        hidden = torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
        reference = _hf_prefill(hf_layer, hf_config, layer_idx, hidden)
        output = decoder.prefill_forward(
            _tt_input(hidden, mesh_device),
            rope_mats=rope,
            page_table=page_table,
            kv_cache=cache,
            valid_seq_len=seq_len,
        )
        _assert_pcc(reference, _to_host(output))


@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
@pytest.mark.parametrize("batch_size", [2, 32])
def test_batched_nonaligned_paged_prefill(hf_config, mesh_device, layer_idx, layer_kind, batch_size):
    seq_len = 33
    state = _layer_state(layer_idx)
    decoder = FunctionalDecoder.from_state_dict(
        state, hf_config=hf_config, layer_idx=layer_idx, mesh_device=mesh_device
    )
    hf_layer = Gemma4TextDecoderLayer(hf_config, layer_idx).to(torch.bfloat16).eval()
    hf_layer.load_state_dict(_local_state(state, layer_idx))
    cache, page_table = _paged_state(hf_config, layer_idx, 128, mesh_device, batch_size=batch_size)
    rope = _rope_device(hf_config, layer_idx, seq_len, mesh_device, decode=False)
    torch.manual_seed(20260900 + layer_idx)
    hidden = torch.randn(batch_size, seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
    reference = _hf_prefill(hf_layer, hf_config, layer_idx, hidden)
    flattened = hidden.reshape(1, batch_size * seq_len, hf_config.hidden_size)
    output = decoder.prefill_forward(
        _tt_input(flattened, mesh_device),
        rope_mats=rope,
        page_table=page_table,
        kv_cache=cache,
        batch_size=batch_size,
        valid_seq_len=seq_len,
    )
    actual = ttnn.to_torch(output).reshape(batch_size, seq_len, hf_config.hidden_size).float()
    _assert_pcc(reference, actual)


@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
def test_batch_32_paged_decode_pcc(hf_config, mesh_device, layer_idx, layer_kind):
    batch_size, prompt_len = 32, 32
    state = _layer_state(layer_idx)
    decoder = FunctionalDecoder.from_state_dict(
        state, hf_config=hf_config, layer_idx=layer_idx, mesh_device=mesh_device
    )
    hf_layer = Gemma4TextDecoderLayer(hf_config, layer_idx).to(torch.bfloat16).eval()
    hf_layer.load_state_dict(_local_state(state, layer_idx))
    cache, page_table = _paged_state(hf_config, layer_idx, 128, mesh_device, batch_size=batch_size)
    prefill_rope = _rope_device(hf_config, layer_idx, prompt_len, mesh_device, decode=False)
    decode_rope = _rope_device(hf_config, layer_idx, 128, mesh_device, decode=True)
    torch.manual_seed(20260950 + layer_idx)
    prompt = torch.randn(batch_size, prompt_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
    token = torch.randn(batch_size, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
    flattened = prompt.reshape(1, batch_size * prompt_len, hf_config.hidden_size)
    decoder.prefill_forward(
        _tt_input(flattened, mesh_device),
        rope_mats=prefill_rope,
        page_table=page_table,
        kv_cache=cache,
        batch_size=batch_size,
        valid_seq_len=prompt_len,
    )

    dynamic_cache = DynamicCache()
    positions = torch.arange(prompt_len).reshape(1, -1)
    with torch.no_grad():
        hf_layer(
            prompt,
            position_embeddings=_rope_host(hf_config, layer_idx, positions),
            attention_mask=torch.triu(torch.full((1, 1, prompt_len, prompt_len), float("-inf")), diagonal=1),
            past_key_values=dynamic_cache,
        )
        reference = hf_layer(
            token,
            position_embeddings=_rope_host(hf_config, layer_idx, [prompt_len]),
            attention_mask=torch.zeros(batch_size, 1, 1, prompt_len + 1),
            past_key_values=dynamic_cache,
        )

    token_for_tt = token.transpose(0, 1).unsqueeze(0)
    tt_token_host = ttnn.from_torch(token_for_tt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_token = ttnn.from_torch(
        token_for_tt,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pos_u_values = torch.full((1, 32), prompt_len, dtype=torch.int32)
    pos_u_host = ttnn.from_torch(pos_u_values, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    pos_u = ttnn.from_torch(
        pos_u_values,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    pos_i_values = torch.full((batch_size,), prompt_len, dtype=torch.int32)
    pos_i_host = ttnn.from_torch(pos_i_values, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    pos_i = ttnn.from_torch(
        pos_i_values,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    def decode_call():
        return decoder.decode_forward(
            tt_token,
            rope_mats=decode_rope,
            page_table=page_table,
            kv_cache=cache,
            current_position=pos_u,
            current_position_cache=pos_i,
            batch_size=batch_size,
        )

    warm = decode_call()
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = decode_call()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        replay_one = ttnn.to_torch(output).reshape(batch_size, 1, hf_config.hidden_size).float()
        _assert_pcc(reference, replay_one)

        # Exercise the production trace contract: update contents of the
        # already-captured token and current-position buffers without replacing
        # their device allocations, then replay the same input deterministically.
        ttnn.copy_host_to_device_tensor(tt_token_host, tt_token)
        ttnn.copy_host_to_device_tensor(pos_u_host, pos_u)
        ttnn.copy_host_to_device_tensor(pos_i_host, pos_i)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        replay_two = ttnn.to_torch(output).reshape(batch_size, 1, hf_config.hidden_size).float()
        assert torch.equal(replay_one, replay_two), "identical batch-32 trace replay was not deterministic"
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
def test_advertised_context_paged_decode(hf_config, mesh_device, layer_idx, layer_kind):
    max_context = hf_config.max_position_embeddings
    state = _layer_state(layer_idx)
    decoder = FunctionalDecoder.from_state_dict(
        state, hf_config=hf_config, layer_idx=layer_idx, mesh_device=mesh_device
    )
    cache, page_table = _paged_state(hf_config, layer_idx, max_context, mesh_device)
    rope = _rope_device(hf_config, layer_idx, max_context, mesh_device, decode=True)
    torch.manual_seed(20261000 + layer_idx)
    token = _tt_input(torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02, mesh_device)
    pos_u_host = torch.zeros((1, 32), dtype=torch.int32)
    pos_u_host[0, 0] = max_context - 1
    pos_u = ttnn.from_torch(pos_u_host, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    pos_i = ttnn.from_torch(
        torch.tensor([max_context - 1], dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    def decode_call():
        return decoder.decode_forward(
            token,
            rope_mats=rope,
            page_table=page_table,
            kv_cache=cache,
            current_position=pos_u,
            current_position_cache=pos_i,
        )

    warm = decode_call()
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = decode_call()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        assert torch.isfinite(_to_host(output)).all()
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.skipif("GEMMA4_LONG_PREFILL" not in os.environ, reason="set GEMMA4_LONG_PREFILL to run capacity probe")
@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
def test_long_nonaligned_prefill_capacity(hf_config, mesh_device, layer_idx, layer_kind):
    seq_len = int(os.environ["GEMMA4_LONG_PREFILL"])
    state = _layer_state(layer_idx)
    decoder = FunctionalDecoder.from_state_dict(
        state, hf_config=hf_config, layer_idx=layer_idx, mesh_device=mesh_device
    )
    hf_layer = Gemma4TextDecoderLayer(hf_config, layer_idx).to(torch.bfloat16).eval()
    hf_layer.load_state_dict(_local_state(state, layer_idx))
    cache_context = math.ceil(seq_len / 128) * 128 if layer_kind == "full_attention" else seq_len
    cache, page_table = _paged_state(hf_config, layer_idx, cache_context, mesh_device)
    rope = _rope_device(hf_config, layer_idx, seq_len, mesh_device, decode=False)
    torch.manual_seed(20261100 + layer_idx)
    hidden = torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
    output = decoder.prefill_forward(
        _tt_input(hidden, mesh_device),
        rope_mats=rope,
        page_table=page_table,
        kv_cache=cache,
        valid_seq_len=seq_len,
    )
    last_output = ttnn.slice(output, [0, 0, seq_len - 1, 0], [1, 1, seq_len, output.shape[-1]])
    assert torch.isfinite(ttnn.to_torch(last_output)).all()

    if layer_kind == "full_attention":
        reference_len = min(seq_len, 2049)
        reference = _hf_prefill(hf_layer, hf_config, layer_idx, hidden[:, :reference_len])
        actual_prefix = ttnn.slice(output, [0, 0, 0, 0], [1, 1, reference_len, output.shape[-1]])
        _assert_pcc(reference, _to_host(actual_prefix))
    else:
        window = hf_config.sliding_window
        start = max(0, seq_len - window)
        reference = _hf_prefill_absolute(hf_layer, hf_config, layer_idx, hidden[:, start:seq_len], start)
        _assert_pcc(reference[:, -1:, :], _to_host(last_output))

    decode_rope = _rope_device(hf_config, layer_idx, seq_len, mesh_device, decode=True)
    decode_token = ttnn.from_torch(
        hidden[:, -1:, :].reshape(1, 1, 1, hf_config.hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pos_u_host = torch.zeros((1, 32), dtype=torch.int32)
    pos_u_host[0, 0] = seq_len - 1
    pos_u = ttnn.from_torch(pos_u_host, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    pos_i = ttnn.from_torch(
        torch.tensor([seq_len - 1], dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    def decode_call():
        return decoder.decode_forward(
            decode_token,
            rope_mats=decode_rope,
            page_table=page_table,
            kv_cache=cache,
            current_position=pos_u,
            current_position_cache=pos_i,
        )

    warm = decode_call()
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    decode_output = decode_call()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        _assert_pcc(_to_host(last_output), _to_host(decode_output))
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.skipif("GEMMA4_PERF" not in os.environ, reason="set GEMMA4_PERF to collect Tracy evidence")
@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
@pytest.mark.parametrize("perf_mode", ["prefill", "decode"])
def test_warmed_performance(hf_config, mesh_device, layer_idx, layer_kind, perf_mode):
    from tracy import signpost

    state = _layer_state(layer_idx)
    decoder = FunctionalDecoder.from_state_dict(
        state, hf_config=hf_config, layer_idx=layer_idx, mesh_device=mesh_device
    )
    cache, page_table = _paged_state(hf_config, layer_idx, 128, mesh_device)
    torch.manual_seed(20261200 + layer_idx)
    if perf_mode == "prefill":
        seq_len = 128
        hidden = _tt_input(torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02, mesh_device)
        rope = _rope_device(hf_config, layer_idx, seq_len, mesh_device, decode=False)

        def call():
            return decoder.prefill_forward(
                hidden, rope_mats=rope, page_table=page_table, kv_cache=cache, valid_seq_len=seq_len
            )

        warm = call()
        ttnn.synchronize_device(mesh_device)
        warm.deallocate(True)
        signpost("PERF_PREFILL")
        output = call()
        ttnn.synchronize_device(mesh_device)
        signpost("PERF_PREFILL_END")
        output.deallocate(True)
        return

    prompt_len = 32
    prompt = _tt_input(torch.randn(1, prompt_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02, mesh_device)
    prefill_rope = _rope_device(hf_config, layer_idx, prompt_len, mesh_device, decode=False)
    decoder.prefill_forward(
        prompt, rope_mats=prefill_rope, page_table=page_table, kv_cache=cache, valid_seq_len=prompt_len
    )
    token = _tt_input(torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02, mesh_device)
    rope = _rope_device(hf_config, layer_idx, 128, mesh_device, decode=True)
    pos_u_host = torch.zeros((1, 32), dtype=torch.int32)
    pos_u_host[0, 0] = prompt_len
    pos_u = ttnn.from_torch(pos_u_host, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    pos_i = ttnn.from_torch(
        torch.tensor([prompt_len], dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    def call():
        return decoder.decode_forward(
            token,
            rope_mats=rope,
            page_table=page_table,
            kv_cache=cache,
            current_position=pos_u,
            current_position_cache=pos_i,
        )

    warm = call()
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = call()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        signpost("PERF_DECODE")
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        signpost("PERF_DECODE_END")
    finally:
        ttnn.release_trace(mesh_device, trace_id)


def test_runtime_fallback_audit():
    source = "\n".join(
        inspect.getsource(method)
        for method in (
            FunctionalDecoder._prefill_attention,
            FunctionalDecoder._streaming_full_prefill_attention,
            FunctionalDecoder._chunked_full_attention,
            FunctionalDecoder._forward_device,
            FunctionalDecoder.prefill_forward,
            FunctionalDecoder.decode_forward,
            Gemma4Attention.__call__,
            gemma_decode_forward,
            RMSNorm.forward,
            SharedMLP.__call__,
        )
    )
    forbidden = ("torch", "ttnn.from_torch", "ttnn.to_torch")
    assert [term for term in forbidden if term in source] == []
