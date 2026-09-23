# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Galaxy correctness tests for Llama-3.1 full-causal GQA and its real-weight composition."""

import json
import math
import os
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from loguru import logger
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

import ttnn
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig
from models.demos.llama_3p1_8b_d_p.tests.device_utils import addresses as _addresses
from models.demos.llama_3p1_8b_d_p.tests.utils import metrics, read_raw_weights
from models.demos.llama_3p1_8b_d_p.tt.attention import AttentionOutputProjection, FullCausalAttention
from models.demos.llama_3p1_8b_d_p.tt.config import MeshConfig
from models.demos.llama_3p1_8b_d_p.tt.kv_cache import allocate_kv_cache, write_kv_chunk
from models.demos.llama_3p1_8b_d_p.tt.qkv import QKVProjection
from models.demos.llama_3p1_8b_d_p.tt.rope import apply_indexed_rope, build_indexed_rope, build_transformation_mat

_metrics = partial(metrics, error_type=ValueError)

HF_MODEL = Path(os.environ.get("LLAMA31_8B_CHECKPOINT", "/mnt/models/meta-llama/Llama-3.1-8B-Instruct"))
MESH_SHAPE = (4, 8)
SP, TP = MESH_SHAPE
SP_AXIS = 0
GLOBAL_CHUNK = 1024
LOCAL_SEQUENCE = GLOBAL_CHUNK // SP
MAX_SEQ_LEN = 2048
NUM_LAYERS = 32
HEAD_DIM = Llama31_8BConfig.HEAD_DIM
HIDDEN_SIZE = Llama31_8BConfig.EMB_SIZE
NUM_Q_HEADS = Llama31_8BConfig.NUM_ATTENTION_HEADS
NUM_KV_HEADS = Llama31_8BConfig.NUM_KEY_VALUE_HEADS
LOCAL_Q_HEADS = NUM_Q_HEADS // TP
RAW_SCENARIOS = (
    (0, 0, 0, 1, 0),
    (1, 13, 0, 33, 1),
    (0, 31, 0, 1024, 0),
    (1, 0, 32, 65, 1),
    (0, 13, 224, 257, 0),
    (1, 31, 288, 1057, 1),
    (0, 0, 768, 1023, 1),
    (1, 13, 1024, 1537, 0),
    (0, 31, 1056, 2048, 1),
    (1, 0, 2016, 2048, 0),
)
WEIGHT_NAMES = {
    "q_proj.weight": "model.layers.0.self_attn.q_proj.weight",
    "k_proj.weight": "model.layers.0.self_attn.k_proj.weight",
    "v_proj.weight": "model.layers.0.self_attn.v_proj.weight",
    "o_proj.weight": "model.layers.0.self_attn.o_proj.weight",
}


def _load_layer_zero_attention_weights():
    return read_raw_weights(HF_MODEL, WEIGHT_NAMES)


def _owned_positions(start):
    owned = [[] for _ in range(SP)]
    for position in range(start, start + GLOBAL_CHUNK):
        owned[(position % GLOBAL_CHUNK) // LOCAL_SEQUENCE].append(position)
    assert all(len(rows) == LOCAL_SEQUENCE for rows in owned)
    return owned


def _device_major_positions(start):
    return [position for rows in _owned_positions(start) for position in rows]


def _fixture(kind, prompt, heads, positions):
    pos = torch.as_tensor(positions, dtype=torch.int64)
    head = torch.arange(heads, dtype=torch.int64)
    dim = torch.arange(HEAD_DIM, dtype=torch.int64)
    if kind == "q":
        code = (pos[None, :, None] * 131 + head[:, None, None] * 977 + dim[None, None, :] * 37 + prompt * 53) % 257
        values = (code.float() - 128) / 256
    elif kind == "k":
        code = (pos[None, :, None] * 193 + head[:, None, None] * 619 + dim[None, None, :] * 71 + prompt * 89) % 263
        values = (code.float() - 131) / 256
    else:
        code = (pos[None, :, None] * 157 + head[:, None, None] * 811 + dim[None, None, :] * 43 + prompt * 101) % 269
        values = (code.float() - 134) / 128
    return values.unsqueeze(0).to(torch.bfloat16).float()


def _positive_baseline_fixture(kind, prompt, heads, positions):
    pos = torch.as_tensor(positions, dtype=torch.int64)
    head = torch.arange(heads, dtype=torch.float32)
    dim = torch.arange(HEAD_DIM, dtype=torch.float32)
    if kind == "q":
        return torch.zeros(1, heads, len(pos), HEAD_DIM)
    base = 0.25 if kind == "k" else 1.0
    values = (
        base
        + head[:, None, None] * 0.25
        + (dim[None, None, :] % 16) * 0.125
        + (pos[None, :, None] % 32).float() * 0.0625
        + prompt * 0.125
    )
    return values.unsqueeze(0).to(torch.bfloat16).float()


def _positive_pulse_fixture(kind, prompt, heads, positions):
    values = _positive_baseline_fixture(kind, prompt, heads, positions)
    if kind != "v":
        return values
    pos = torch.as_tensor(positions, dtype=torch.int64)
    head = torch.arange(heads, dtype=torch.float32)
    dim = torch.arange(HEAD_DIM, dtype=torch.float32)
    for pulse_position, amplitude in zip((32, 256, 1024, 2047), (16.0, 64.0, 256.0, 512.0)):
        pulse = amplitude * (1.0 + head[:, None, None] * 0.015625 + (dim[None, None, :] % 8) * 0.0078125)
        values += (pos == pulse_position)[None, :, None] * pulse
    return values.to(torch.bfloat16).float()


def _physical_fixture(kind, prompt, heads, start, fixture_fn=_fixture):
    return fixture_fn(kind, prompt, heads, _device_major_positions(start))


def _reference_attention(q, k, v, query_positions):
    outputs = []
    key_positions = torch.arange(k.shape[2])
    scale = HEAD_DIM**-0.5
    for q_head in range(NUM_Q_HEADS):
        kv_head = q_head // (NUM_Q_HEADS // NUM_KV_HEADS)
        scores = torch.matmul(q[0, q_head].float(), k[0, kv_head].float().transpose(0, 1))
        scores = scores * scale
        scores = scores.masked_fill(
            key_positions[None, :] > query_positions[:, None],
            float("-inf"),
        )
        probs = torch.softmax(scores, dim=-1)
        head_output = torch.matmul(probs, v[0, kv_head].float())
        outputs.append(head_output)
    return torch.stack(outputs, dim=0).unsqueeze(0)


def _reference_fixture(
    prompt,
    start,
    end,
    *,
    zero_qk=False,
    zero_q=False,
    zero_v=False,
    fixture_fn=_fixture,
):
    query_positions = torch.arange(start, end)
    q = fixture_fn("q", prompt, NUM_Q_HEADS, query_positions)
    k = fixture_fn("k", prompt, NUM_KV_HEADS, range(end))
    v = fixture_fn("v", prompt, NUM_KV_HEADS, range(end))
    if zero_qk:
        q = torch.zeros_like(q)
        k = torch.zeros_like(k)
    elif zero_q:
        q = torch.zeros_like(q)
    if zero_v:
        v = torch.zeros_like(v)
    return _reference_attention(q, k, v, query_positions)


def _prefix_average_fixture(prompt, start, end, fixture_fn=_fixture):
    v = fixture_fn("v", prompt, NUM_KV_HEADS, range(end))
    prefix_sum = torch.cumsum(v, dim=2)
    denominator = torch.arange(1, end + 1, dtype=torch.float32)[None, None, :, None]
    kv_output = prefix_sum / denominator
    return kv_output.repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, dim=1)[:, :, start:end]


def _to_q(mesh_device, values, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        values.to(torch.bfloat16),
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(2, 1)),
    )


def _to_kv(mesh_device, values):
    return _to_q(mesh_device, values)


def _persistent_attention_addresses(attention):
    return {
        name: _addresses(getattr(attention, name))
        for name in ("gathered_k", "gathered_v", "query_position_table", "key_positions")
    }


def _write_prefix(
    mesh_device,
    cache,
    *,
    slot,
    layer,
    end,
    prompt,
    zero_qk=False,
    zero_v=False,
    dirty_after=None,
    fixture_fn=_fixture,
):
    inputs = []
    for chunk_start in range(0, end, GLOBAL_CHUNK):
        positions = _device_major_positions(chunk_start)
        k = fixture_fn("k", prompt, NUM_KV_HEADS, positions)
        v = fixture_fn("v", prompt, NUM_KV_HEADS, positions)
        if zero_qk:
            k.zero_()
        if zero_v:
            v.zero_()
        if dirty_after is not None:
            dirty = torch.tensor(positions) >= dirty_after
            k[:, :, dirty, :] = 64.0
            v[:, :, dirty, :] = -96.0
        tt_k = _to_kv(mesh_device, k)
        tt_v = _to_kv(mesh_device, v)
        write_kv_chunk(
            cache,
            tt_k,
            tt_v,
            slot_idx=slot,
            layer_idx=layer,
            actual_start=chunk_start,
            actual_end=min(end, chunk_start + GLOBAL_CHUNK),
        )
        inputs.extend((tt_k, tt_v))
    return inputs


def _read_cache_prefix(cache, batch_index, end):
    cache_shards = {
        name: [ttnn.to_torch(shard)[batch_index, 0].float() for shard in ttnn.get_device_tensors(tensor)]
        for name, tensor in (("k", cache.k), ("v", cache.v))
    }
    natural = {}
    for name, shards in cache_shards.items():
        value = torch.empty(1, NUM_KV_HEADS, end, HEAD_DIM)
        for position in range(end):
            sp_coord = (position % GLOBAL_CHUNK) // LOCAL_SEQUENCE
            local_row = (position // GLOBAL_CHUNK) * LOCAL_SEQUENCE + position % LOCAL_SEQUENCE
            for kv_head in range(NUM_KV_HEADS):
                value[0, kv_head, position] = shards[sp_coord * TP + kv_head][local_row]
        natural[name] = value
    return natural["k"], natural["v"]


def _read_query_interval(q, start, end):
    """Restore valid Q rows to natural order and convert Meta coordinates to HF coordinates."""
    shards = ttnn.get_device_tensors(q)
    assert len(shards) == SP * TP
    natural = torch.full((1, NUM_Q_HEADS, end - start, HEAD_DIM), float("nan"), dtype=torch.float32)
    for sp_coord, positions in enumerate(_owned_positions(start)):
        local_rows = [row for row, position in enumerate(positions) if position < end]
        natural_rows = [position - start for position in positions if position < end]
        if not local_rows:
            continue
        for tp_coord in range(TP):
            actual = ttnn.to_torch(shards[sp_coord * TP + tp_coord]).float()[:, :LOCAL_Q_HEADS, local_rows, :HEAD_DIM]
            h0 = tp_coord * LOCAL_Q_HEADS
            natural[:, h0 : h0 + LOCAL_Q_HEADS, natural_rows, :] = torch.cat(
                (actual[..., 0::2], actual[..., 1::2]), dim=-1
            )
    assert torch.isfinite(natural).all(), "query readback contains non-finite or missing valid rows"
    return natural


def _diagnostic_metrics(expected, actual):
    pcc, nl2 = _metrics(expected, actual)
    error = actual.double() - expected.double()
    metrics = {
        "pcc": pcc,
        "nl2": nl2,
        "expected_rms": torch.sqrt(torch.mean(expected.double().square())).item(),
        "error_rms": torch.sqrt(torch.mean(error.square())).item(),
        "max_abs": torch.max(torch.abs(error)).item(),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError(f"diagnostic metrics must be finite, got {metrics}")
    return metrics


def _cache_plane_snapshots(cache, batch_index):
    return [
        [ttnn.to_torch(shard)[batch_index : batch_index + 1].clone() for shard in ttnn.get_device_tensors(tensor)]
        for tensor in (cache.k, cache.v)
    ]


def _assert_cache_plane_unchanged(cache, batch_index, before):
    for tensor, snapshots in zip((cache.k, cache.v), before):
        for shard, snapshot in zip(ttnn.get_device_tensors(tensor), snapshots):
            assert torch.equal(ttnn.to_torch(shard)[batch_index : batch_index + 1], snapshot)


def _run_attention_case(
    mesh_device,
    attention,
    cache,
    *,
    slot,
    layer,
    start,
    end,
    prompt,
    cache_dtype,
    label,
    zero_qk=False,
    zero_q=False,
    zero_v=False,
    fixture_fn=_fixture,
    enforce_metrics=True,
):
    prefix_inputs = _write_prefix(
        mesh_device,
        cache,
        slot=slot,
        layer=layer,
        end=end,
        prompt=prompt,
        zero_qk=zero_qk,
        zero_v=zero_v,
        fixture_fn=fixture_fn,
    )
    q_values = _physical_fixture("q", prompt, NUM_Q_HEADS, start, fixture_fn)
    if zero_qk or zero_q:
        q_values.zero_()
    q = _to_q(mesh_device, q_values)
    q_before = [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(q)]
    batch_index = slot * NUM_LAYERS + layer
    cache_before = _cache_plane_snapshots(cache, batch_index)
    output = attention(q, cache, slot_idx=slot, layer_idx=layer, actual_start=start, actual_end=end)
    ttnn.synchronize_device(mesh_device)

    assert tuple(output.shape) == (1, LOCAL_Q_HEADS, LOCAL_SEQUENCE, HEAD_DIM)
    assert output.dtype == ttnn.bfloat16
    assert output.layout == ttnn.TILE_LAYOUT
    assert output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    for shard, snapshot in zip(ttnn.get_device_tensors(q), q_before):
        assert torch.equal(ttnn.to_torch(shard), snapshot)
    _assert_cache_plane_unchanged(cache, batch_index, cache_before)

    if (zero_q or zero_qk) and not zero_v:
        expected = _prefix_average_fixture(prompt, start, end, fixture_fn)
        softmax_expected = _reference_fixture(
            prompt,
            start,
            end,
            zero_qk=zero_qk,
            zero_q=zero_q,
            fixture_fn=fixture_fn,
        )
        assert torch.allclose(expected, softmax_expected, atol=2e-6, rtol=2e-5)
    else:
        expected = _reference_fixture(
            prompt,
            start,
            end,
            zero_qk=zero_qk,
            zero_q=zero_q,
            zero_v=zero_v,
            fixture_fn=fixture_fn,
        )
    owned = _owned_positions(start)
    output_shards = ttnn.get_device_tensors(output)
    per_device = {}
    errors = []
    pcc_limit, nl2_limit = (0.999, 0.02) if cache_dtype == ttnn.bfloat16 else (0.999, 0.03)
    for sp_coord in range(SP):
        valid_rows = [row for row, position in enumerate(owned[sp_coord]) if position < end]
        expected_rows = [position - start for position in owned[sp_coord] if position < end]
        for tp_coord in range(TP):
            device_idx = sp_coord * TP + tp_coord
            h0 = tp_coord * LOCAL_Q_HEADS
            actual = ttnn.to_torch(output_shards[device_idx]).float()[:, :LOCAL_Q_HEADS, valid_rows, :HEAD_DIM]
            wanted = expected[:, h0 : h0 + LOCAL_Q_HEADS, expected_rows, :]
            per_device[device_idx] = actual.clone()
            assert torch.isfinite(actual).all(), f"{label} chip={device_idx}: non-finite output"
            assert torch.isfinite(wanted).all(), f"{label} chip={device_idx}: non-finite reference"
            if not valid_rows:
                continue
            if zero_v:
                assert torch.count_nonzero(actual) == 0
                pcc, nl2 = 1.0, 0.0
            else:
                pcc, nl2 = _metrics(wanted, actual)
                if enforce_metrics:
                    assert pcc >= pcc_limit, f"{label} chip={device_idx}: PCC={pcc:.7f}, NL2={nl2:.7f}"
                if enforce_metrics:
                    assert nl2 <= nl2_limit, f"{label} chip={device_idx}: PCC={pcc:.7f}, NL2={nl2:.7f}"
            errors.append((pcc, nl2))
    assert errors, f"{label}: no valid rows were evaluated"
    logger.info(
        f"attention {label}: dtype={cache_dtype}, start={start}, end={end}, "
        f"min_PCC={min(x[0] for x in errors):.7f}, max_NL2={max(x[1] for x in errors):.7f}, "
        f"q_addresses={[hex(x) for x in _addresses(q)]}"
    )
    output.deallocate(True)
    q.deallocate(True)
    for tensor in prefix_inputs:
        tensor.deallocate(True)
    return per_device


def _assert_attention_readiness(mesh_device, attention, cache, cache_dtype):
    assert cache_dtype == attention.cache_dtype
    assert tuple(attention.gathered_k.shape) == (1, 1, MAX_SEQ_LEN, HEAD_DIM)
    assert tuple(attention.gathered_v.shape) == (1, 1, MAX_SEQ_LEN, HEAD_DIM)
    assert set(_addresses(attention.gathered_k)).isdisjoint(_addresses(attention.gathered_v))
    assert attention.program_config.exp_approx_mode is False
    assert attention.compute_kernel_config.fp32_dest_acc_en is True
    grid = mesh_device.compute_with_storage_grid_size()
    assert attention.program_config.compute_with_storage_grid_size == ttnn.CoreCoord(grid.x - 1, grid.y)
    edges = []
    for tp_coord in range(TP):
        for sp_coord in range(SP):
            src_coord = ttnn.MeshCoordinate([sp_coord, tp_coord])
            src_node = mesh_device.get_fabric_node_id(src_coord)
            for neighbor_sp in ((sp_coord - 1) % SP, (sp_coord + 1) % SP):
                dst_coord = ttnn.MeshCoordinate([neighbor_sp, tp_coord])
                dst_node = mesh_device.get_fabric_node_id(dst_coord)
                links = tuple(ttnn.get_forwarding_link_indices(src_node, dst_node))
                assert 0 in links
                edges.append((src_coord, dst_coord, links))
    assert len(edges) == 64
    attention.validate_request(cache, slot_idx=0, layer_idx=0, actual_start=0, actual_end=1)
    l1 = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1)
    assert l1.total_bytes_free_per_bank >= attention._SDPA_L1_BYTES
    assert l1.largest_contiguous_bytes_free_per_bank >= attention._SDPA_L1_BYTES
    logger.info(
        f"attention readiness: mesh={tuple(mesh_device.shape)}, grid={grid}, "
        f"sdpa_grid={attention.program_config.compute_with_storage_grid_size}, "
        f"SP_edges={edges}, L1_free={l1.total_bytes_free_per_bank}, "
        f"L1_largest={l1.largest_contiguous_bytes_free_per_bank}, "
        f"required={attention._SDPA_L1_BYTES}"
    )


def _composition_hidden(prompt=0):
    generator = torch.Generator().manual_seed(2026091601 + prompt)
    values = torch.randn(1, 1, MAX_SEQ_LEN, HIDDEN_SIZE, generator=generator) * 0.125
    values += (torch.arange(MAX_SEQ_LEN, dtype=torch.float32)[None, None, :, None] + prompt * 7) % 17 / 512
    return values


def _real_token_hiddens(stream_mode):
    prompt_texts = (
        "The capital of France is",
        "Give a concise proof that the square root of two is irrational.",
    )
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, local_files_only=True)
    token_streams = {}
    for prompt, text in enumerate(prompt_texts):
        if stream_mode == "repeated_bos":
            token_ids = tokenizer.encode(text, add_special_tokens=True)
            token_streams[prompt] = (token_ids * math.ceil(MAX_SEQ_LEN / len(token_ids)))[:MAX_SEQ_LEN]
        elif stream_mode == "bos_once":
            repeated_text = text
            token_ids = tokenizer.encode(repeated_text, add_special_tokens=True)
            while len(token_ids) < MAX_SEQ_LEN:
                repeated_text = f"{repeated_text} {repeated_text}"
                token_ids = tokenizer.encode(repeated_text, add_special_tokens=True)
            token_streams[prompt] = token_ids[:MAX_SEQ_LEN]
        else:
            raise ValueError(f"unknown real-token stream mode: {stream_mode}")

    with (HF_MODEL / "model.safetensors.index.json").open() as index_file:
        weight_map = json.load(index_file)["weight_map"]
    embedding_name = "model.embed_tokens.weight"
    norm_name = "model.layers.0.input_layernorm.weight"
    unique_ids = sorted({token_id for stream in token_streams.values() for token_id in stream})
    with safe_open(HF_MODEL / weight_map[embedding_name], framework="pt", device="cpu") as checkpoint:
        embedding_slice = checkpoint.get_slice(embedding_name)
        rows = {token_id: embedding_slice[token_id : token_id + 1].squeeze(0) for token_id in unique_ids}
    with safe_open(HF_MODEL / weight_map[norm_name], framework="pt", device="cpu") as checkpoint:
        norm_weight = checkpoint.get_tensor(norm_name).float()

    hiddens = {}
    metadata = {}
    for prompt, stream in token_streams.items():
        embeddings = torch.stack([rows[token_id] for token_id in stream]).float()
        normalized = embeddings * torch.rsqrt(torch.mean(embeddings.square(), dim=-1, keepdim=True) + 1e-5)
        normalized *= norm_weight
        hiddens[prompt] = normalized.reshape(1, 1, MAX_SEQ_LEN, HIDDEN_SIZE)
        metadata[str(prompt)] = {
            "text": prompt_texts[prompt],
            "stream_mode": stream_mode,
            "base_token_ids": tokenizer.encode(prompt_texts[prompt], add_special_tokens=True),
            "rms": torch.sqrt(torch.mean(normalized.square())).item(),
            "max_abs": torch.max(torch.abs(normalized)).item(),
        }
    return hiddens, metadata


def _to_hidden_chunk(mesh_device, hidden, start):
    positions = torch.tensor(_device_major_positions(start))
    valid = positions < MAX_SEQ_LEN
    physical = torch.zeros(1, 1, GLOBAL_CHUNK, HIDDEN_SIZE, dtype=hidden.dtype)
    physical[:, :, valid, :] = hidden[:, :, positions[valid], :]
    return ttnn.from_torch(
        physical.to(torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(2, None)),
    )


def _rotate_half(tensor):
    half = tensor.shape[-1] // 2
    return torch.cat((-tensor[..., half:], tensor[..., :half]), dim=-1)


def _hf_rotate(rotary, tensor, positions):
    position_ids = torch.tensor([positions], dtype=torch.long)
    cos, sin = rotary(tensor, position_ids)
    return tensor * cos[:, None, :, :] + _rotate_half(tensor) * sin[:, None, :, :]


def _composition_reference_state(hidden, weights, rotary):
    rounded = hidden.to(torch.bfloat16).float()
    q = F.linear(rounded, weights["q_proj.weight"].to(torch.bfloat16).float())
    k = F.linear(rounded, weights["k_proj.weight"].to(torch.bfloat16).float())
    v = F.linear(rounded, weights["v_proj.weight"].to(torch.bfloat16).float())
    q = q.reshape(1, MAX_SEQ_LEN, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
    k = k.reshape(1, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    v = v.reshape(1, MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    positions = range(MAX_SEQ_LEN)
    return {
        "q": _hf_rotate(rotary, q, positions),
        "k": _hf_rotate(rotary, k, positions),
        "v": v,
    }


def _composition_expected_from_state(state, weights, *, start, end):
    q = state["q"][:, :, start:end]
    heads = _reference_attention(
        q,
        state["k"][:, :, :end],
        state["v"][:, :, :end],
        torch.arange(start, end),
    )
    concatenated = heads.transpose(1, 2).reshape(1, 1, end - start, HIDDEN_SIZE)
    output = F.linear(concatenated, weights["o_proj.weight"].to(torch.bfloat16).float())
    return q, heads, output


def _project_rotate_write(
    mesh_device,
    qkv,
    rope_tables,
    transformation,
    cache,
    hidden,
    *,
    slot,
    layer,
    start,
    end,
):
    tt_hidden = _to_hidden_chunk(mesh_device, hidden, start)
    hidden_before = [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(tt_hidden)]
    q, k, v = qkv(tt_hidden)
    q_rot = apply_indexed_rope(q, rope_tables, transformation, kv_actual_global=start, sp_axis=SP_AXIS)
    k_rot = apply_indexed_rope(k, rope_tables, transformation, kv_actual_global=start, sp_axis=SP_AXIS)
    write_kv_chunk(
        cache,
        k_rot,
        v,
        slot_idx=slot,
        layer_idx=layer,
        actual_start=start,
        actual_end=end,
    )
    ttnn.synchronize_device(mesh_device)
    for shard, snapshot in zip(ttnn.get_device_tensors(tt_hidden), hidden_before):
        assert torch.equal(ttnn.to_torch(shard), snapshot)
    q.deallocate(True)
    k.deallocate(True)
    v.deallocate(True)
    k_rot.deallocate(True)
    tt_hidden.deallocate(True)
    return q_rot


def _run_composition_case(
    mesh_device,
    qkv,
    rope_tables,
    transformation,
    attention,
    output_projection,
    cache,
    hidden,
    expected,
    *,
    slot,
    layer,
    start,
    end,
    cache_dtype,
    label,
    populate_prefix=True,
    reference_heads=None,
    reference_q=None,
    reference_state=None,
    enforce_metrics=True,
    return_metrics=False,
):
    assert torch.isfinite(expected).all(), f"{label}: non-finite output reference"
    if reference_heads is not None:
        assert torch.isfinite(reference_heads).all(), f"{label}: non-finite source-head reference"
    if reference_q is not None:
        assert torch.isfinite(reference_q).all(), f"{label}: non-finite source-Q reference"
    if populate_prefix and start:
        prefix_start = 0
        while prefix_start < start:
            prefix_end = min(start, prefix_start + GLOBAL_CHUNK)
            prefix_q = _project_rotate_write(
                mesh_device,
                qkv,
                rope_tables,
                transformation,
                cache,
                hidden,
                slot=slot,
                layer=layer,
                start=prefix_start,
                end=prefix_end,
            )
            prefix_q.deallocate(True)
            prefix_start += GLOBAL_CHUNK
    q_rot = _project_rotate_write(
        mesh_device,
        qkv,
        rope_tables,
        transformation,
        cache,
        hidden,
        slot=slot,
        layer=layer,
        start=start,
        end=end,
    )
    heads = attention(q_rot, cache, slot_idx=slot, layer_idx=layer, actual_start=start, actual_end=end)
    cache_reference_heads = None
    if reference_q is not None:
        batch_index = slot * NUM_LAYERS + layer
        readback_k, readback_v = _read_cache_prefix(cache, batch_index, end)
        # Indexed RoPE stores cache K in Meta's interleaved frame; the independent HF reference Q
        # is half-split, so convert cache readback before computing the attribution-only head oracle.
        readback_k = torch.cat((readback_k[..., 0::2], readback_k[..., 1::2]), dim=-1)
        cache_reference_heads = _reference_attention(
            reference_q,
            readback_k,
            readback_v,
            torch.arange(start, end),
        )
        assert torch.isfinite(cache_reference_heads).all(), f"{label}: non-finite cache-readback reference"
    exact_input_heads = None
    if reference_state is not None:
        assert reference_q is not None and reference_heads is not None
        readback_q = _read_query_interval(q_rot, start, end)
        # Recompute CPU SDPA on the exact device Q and stored K/V. The ideal-Q oracle above
        # remains attribution-only: its device comparison includes Q rounding error as well as SDPA error.
        exact_input_heads = _reference_attention(readback_q, readback_k, readback_v, torch.arange(start, end))
        assert torch.isfinite(exact_input_heads).all(), f"{label}: non-finite exact-input reference"
    output = output_projection(heads)
    ttnn.synchronize_device(mesh_device)

    owned = _owned_positions(start)
    head_shards = ttnn.get_device_tensors(heads)
    output_shards = ttnn.get_device_tensors(output)
    errors = []
    per_chip = {}
    if reference_state is not None:
        # Check the whole stored prefix, including SP rows without valid Q in a partial continuation.
        for sp_coord in range(SP):
            cache_positions = [
                position for position in range(end) if (position % GLOBAL_CHUNK) // LOCAL_SEQUENCE == sp_coord
            ]
            if not cache_positions:
                continue
            for tp_coord in range(TP):
                cache_slice = (slice(None), slice(tp_coord, tp_coord + 1), cache_positions, slice(None))
                per_chip[str(sp_coord * TP + tp_coord)] = {
                    "source_cache_k": _diagnostic_metrics(reference_state["k"][cache_slice], readback_k[cache_slice]),
                    "source_cache_v": _diagnostic_metrics(reference_state["v"][cache_slice], readback_v[cache_slice]),
                }
    pcc_limit, nl2_limit = (0.999, 0.03) if cache_dtype == ttnn.bfloat16 else (0.995, 0.05)
    for sp_coord in range(SP):
        valid_rows = [row for row, position in enumerate(owned[sp_coord]) if position < end]
        expected_rows = [position - start for position in owned[sp_coord] if position < end]
        if not valid_rows:
            continue
        tp_outputs = []
        wanted = expected[:, :, expected_rows, :]
        assert torch.isfinite(wanted).all(), f"{label} SP={sp_coord}: non-finite output reference"
        for tp_coord in range(TP):
            device_idx = sp_coord * TP + tp_coord
            h0 = tp_coord * LOCAL_Q_HEADS
            actual_heads = ttnn.to_torch(head_shards[device_idx]).float()[:, :LOCAL_Q_HEADS, valid_rows, :HEAD_DIM]
            assert torch.isfinite(actual_heads).all(), f"{label} chip={device_idx}: non-finite heads"
            actual = ttnn.to_torch(output_shards[device_idx]).float()[:, :, valid_rows, :HIDDEN_SIZE]
            assert torch.isfinite(actual).all(), f"{label} chip={device_idx}: non-finite output"
            tp_outputs.append(actual)
            pcc, nl2 = _metrics(wanted, actual)
            if enforce_metrics:
                assert pcc >= pcc_limit, f"{label} chip={device_idx}: PCC={pcc:.7f}, NL2={nl2:.7f}"
                assert nl2 <= nl2_limit, f"{label} chip={device_idx}: PCC={pcc:.7f}, NL2={nl2:.7f}"
            chip_metrics = per_chip.setdefault(str(device_idx), {})
            chip_metrics["output"] = _diagnostic_metrics(wanted, actual)
            if reference_heads is not None:
                source_heads = reference_heads[:, h0 : h0 + LOCAL_Q_HEADS, expected_rows]
                assert torch.isfinite(source_heads).all(), f"{label} chip={device_idx}: non-finite source heads"
                chip_metrics["source_heads"] = _diagnostic_metrics(source_heads, actual_heads)
            if cache_reference_heads is not None:
                readback_heads = cache_reference_heads[:, h0 : h0 + LOCAL_Q_HEADS, expected_rows]
                assert torch.isfinite(
                    readback_heads
                ).all(), f"{label} chip={device_idx}: non-finite cache-readback heads"
                chip_metrics["cache_readback_heads"] = _diagnostic_metrics(readback_heads, actual_heads)
                if reference_heads is not None:
                    chip_metrics["head_reference_delta"] = _diagnostic_metrics(source_heads, readback_heads)
            if exact_input_heads is not None:
                q_slice = (slice(None), slice(h0, h0 + LOCAL_Q_HEADS), expected_rows, slice(None))
                chip_metrics["rotated_q"] = _diagnostic_metrics(reference_q[q_slice], readback_q[q_slice])
                exact_heads = exact_input_heads[:, h0 : h0 + LOCAL_Q_HEADS, expected_rows]
                chip_metrics["exact_input_heads"] = _diagnostic_metrics(exact_heads, actual_heads)
                chip_metrics["exact_input_reference_delta"] = _diagnostic_metrics(source_heads, exact_heads)
            per_chip[str(device_idx)] = chip_metrics
            assert all(
                math.isfinite(value) for stage_metrics in chip_metrics.values() for value in stage_metrics.values()
            ), f"{label} chip={device_idx}: non-finite derived metrics {chip_metrics}"
            errors.append((pcc, nl2))
        for replicated in tp_outputs[1:]:
            assert torch.equal(replicated, tp_outputs[0]), f"{label} SP={sp_coord}: TP outputs differ"
    assert errors, f"{label}: no valid composition rows were evaluated"
    logger.info(
        f"attention composition {label}: dtype={cache_dtype}, min_PCC={min(x[0] for x in errors):.7f}, "
        f"max_NL2={max(x[1] for x in errors):.7f}"
    )
    output.deallocate(True)
    heads.deallocate(True)
    q_rot.deallocate(True)
    if return_metrics:
        references = (
            "output",
            "source_heads",
            "cache_readback_heads",
            "head_reference_delta",
            "rotated_q",
            "source_cache_k",
            "source_cache_v",
            "exact_input_heads",
            "exact_input_reference_delta",
        )
        summary = {}
        for reference in references:
            values = [chip[reference] for chip in per_chip.values() if reference in chip]
            if values:
                summary[reference] = {
                    "min_pcc": min(value["pcc"] for value in values),
                    "max_nl2": max(value["nl2"] for value in values),
                    "max_error_rms": max(value["error_rms"] for value in values),
                    "max_abs": max(value["max_abs"] for value in values),
                }
                assert all(math.isfinite(value) for value in summary[reference].values())
        return {"summary": summary, "per_chip": per_chip}


# Run real layer-0 weights over every raw-cache interval and both seeded prompts for BF16 and BF8,
# retaining the original composition gates while recording attention heads before O projection against
# source and cache-readback references; this separates functional coverage from cancelling stress data.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(MESH_SHAPE, id="galaxy-4x8")], indirect=True)
def test_real_weight_attention_composition_all_raw_scenarios(mesh_device):
    artifact_path = os.environ.get("LLAMA_PREFILL_ATTENTION_WEIGHT_METRICS")
    mesh_config = MeshConfig(MESH_SHAPE, TP)
    weights = _load_layer_zero_attention_weights()
    hiddens = {prompt: _composition_hidden(prompt) for prompt in (0, 1)}
    hf_config = AutoConfig.from_pretrained(HF_MODEL)
    rotary = LlamaRotaryEmbedding(hf_config)
    states = {prompt: _composition_reference_state(hidden, weights, rotary) for prompt, hidden in hiddens.items()}
    qkv = QKVProjection(mesh_device, mesh_config, weights)
    output_projection = AttentionOutputProjection(mesh_device, mesh_config, weights)
    rope_tables = build_indexed_rope(
        mesh_device,
        max_seq_len=MAX_SEQ_LEN,
        chunk_size=GLOBAL_CHUNK,
        sp_axis=SP_AXIS,
    )
    transformation = build_transformation_mat(mesh_device)
    all_results = {}
    failures = []

    for cache_dtype in (ttnn.bfloat16, ttnn.bfloat8_b):
        dtype_name = "bfloat16" if cache_dtype == ttnn.bfloat16 else "bfloat8_b"
        attention = FullCausalAttention(mesh_device, mesh_config, cache_dtype=cache_dtype)
        cache = allocate_kv_cache(mesh_device, mesh_config, cache_dtype=cache_dtype)
        _assert_attention_readiness(mesh_device, attention, cache, cache_dtype)
        dtype_results = []
        pcc_limit, nl2_limit = (0.999, 0.03) if cache_dtype == ttnn.bfloat16 else (0.995, 0.05)

        for slot, layer, start, end, prompt in RAW_SCENARIOS:
            reference_q, reference_heads, expected = _composition_expected_from_state(
                states[prompt],
                weights,
                start=start,
                end=end,
            )
            label = f"real-all-{dtype_name}-s{slot}-l{layer}-{start}-{end}-p{prompt}"
            metrics = _run_composition_case(
                mesh_device,
                qkv,
                rope_tables,
                transformation,
                attention,
                output_projection,
                cache,
                hiddens[prompt],
                expected,
                slot=slot,
                layer=layer,
                start=start,
                end=end,
                cache_dtype=cache_dtype,
                label=label,
                reference_heads=reference_heads,
                reference_q=reference_q,
                enforce_metrics=False,
                return_metrics=True,
            )
            case = {
                "slot": slot,
                "layer": layer,
                "start": start,
                "end": end,
                "prompt": prompt,
                **metrics,
            }
            dtype_results.append(case)
            all_results[dtype_name] = dtype_results
            for stage in ("source_heads", "output"):
                stage_summary = metrics["summary"][stage]
                if stage_summary["min_pcc"] < pcc_limit or stage_summary["max_nl2"] > nl2_limit:
                    failures.append(
                        f"{label}/{stage}: PCC={stage_summary['min_pcc']:.7f}, " f"NL2={stage_summary['max_nl2']:.7f}"
                    )
            if artifact_path:
                with Path(artifact_path).open("w") as artifact_file:
                    json.dump(all_results, artifact_file, indent=2)
            logger.info(f"attention real all metrics {dtype_name}: {case}")

    assert not failures, "real-weight composition gate failures: " + "; ".join(failures)


# Derive repeated-BOS stress and BOS-once normal token streams from real checkpoint embedding rows and
# layer-0 RMSNorm, then verify first-chunk and continuation heads/output in both cache dtypes.
# Gate rotated Q and stored K/V against the source, and SDPA against an independent CPU oracle
# using the exact device Q/K/V. Source-head PCC and source-output PCC/NL2 remain enforced.
# BF8 source-head NL2 is characterization; the BF16 limit remains enforced. Upstream rounding
# can exceed 5% before SDPA, while less accurate SDPA can appear closer through cancellation.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(MESH_SHAPE, id="galaxy-4x8")], indirect=True)
def test_real_token_derived_attention_composition(mesh_device):
    artifact_path = os.environ.get("LLAMA_PREFILL_ATTENTION_TOKEN_METRICS")
    mesh_config = MeshConfig(MESH_SHAPE, TP)
    weights = _load_layer_zero_attention_weights()
    hf_config = AutoConfig.from_pretrained(HF_MODEL)
    rotary = LlamaRotaryEmbedding(hf_config)
    hiddens = {}
    states = {}
    prompt_metadata = {}
    for stream_mode in ("repeated_bos", "bos_once"):
        mode_hiddens, mode_metadata = _real_token_hiddens(stream_mode)
        for prompt, hidden in mode_hiddens.items():
            key = (stream_mode, prompt)
            hiddens[key] = hidden
            states[key] = _composition_reference_state(hidden, weights, rotary)
            prompt_metadata[f"{stream_mode}-{prompt}"] = mode_metadata[str(prompt)]
    qkv = QKVProjection(mesh_device, mesh_config, weights)
    output_projection = AttentionOutputProjection(mesh_device, mesh_config, weights)
    rope_tables = build_indexed_rope(
        mesh_device,
        max_seq_len=MAX_SEQ_LEN,
        chunk_size=GLOBAL_CHUNK,
        sp_axis=SP_AXIS,
    )
    transformation = build_transformation_mat(mesh_device)
    scenarios = (
        ("repeated_bos", 0, 0, 0, 1024, 0, True),
        ("repeated_bos", 0, 0, 1024, 1537, 0, False),
        ("repeated_bos", 1, 13, 1056, 2048, 1, True),
        ("bos_once", 0, 0, 0, 1024, 0, True),
        ("bos_once", 0, 0, 1024, 1537, 0, False),
        ("bos_once", 1, 13, 1056, 2048, 1, True),
    )
    all_results = {"prompts": prompt_metadata}
    failures = []

    for cache_dtype in (ttnn.bfloat16, ttnn.bfloat8_b):
        dtype_name = "bfloat16" if cache_dtype == ttnn.bfloat16 else "bfloat8_b"
        attention = FullCausalAttention(mesh_device, mesh_config, cache_dtype=cache_dtype)
        cache = allocate_kv_cache(mesh_device, mesh_config, cache_dtype=cache_dtype)
        _assert_attention_readiness(mesh_device, attention, cache, cache_dtype)
        dtype_results = []
        pcc_limit, nl2_limit = (0.999, 0.03) if cache_dtype == ttnn.bfloat16 else (0.995, 0.05)

        for stream_mode, slot, layer, start, end, prompt, populate_prefix in scenarios:
            key = (stream_mode, prompt)
            reference_q, reference_heads, expected = _composition_expected_from_state(
                states[key],
                weights,
                start=start,
                end=end,
            )
            label = f"real-token-{stream_mode}-{dtype_name}-s{slot}-l{layer}-{start}-{end}-p{prompt}"
            metrics = _run_composition_case(
                mesh_device,
                qkv,
                rope_tables,
                transformation,
                attention,
                output_projection,
                cache,
                hiddens[key],
                expected,
                slot=slot,
                layer=layer,
                start=start,
                end=end,
                cache_dtype=cache_dtype,
                label=label,
                populate_prefix=populate_prefix,
                reference_heads=reference_heads,
                reference_q=reference_q,
                reference_state=states[key],
                enforce_metrics=False,
                return_metrics=True,
            )
            case = {
                "slot": slot,
                "layer": layer,
                "start": start,
                "end": end,
                "prompt": prompt,
                "stream_mode": stream_mode,
                **metrics,
            }
            dtype_results.append(case)
            all_results[dtype_name] = dtype_results
            cache_limits = (0.9999, 0.01) if cache_dtype == ttnn.bfloat16 else (0.999, 0.02)
            stage_limits = {
                "source_heads": (pcc_limit, nl2_limit if cache_dtype == ttnn.bfloat16 else None),
                "output": (pcc_limit, nl2_limit),
                "rotated_q": (0.9999, 0.01),
                "source_cache_k": cache_limits,
                "source_cache_v": cache_limits,
                "exact_input_heads": (0.9999, 0.01),
            }
            for stage, (stage_pcc_limit, stage_nl2_limit) in stage_limits.items():
                stage_summary = metrics["summary"][stage]
                if stage_summary["min_pcc"] < stage_pcc_limit or (
                    stage_nl2_limit is not None and stage_summary["max_nl2"] > stage_nl2_limit
                ):
                    failures.append(
                        f"{label}/{stage}: PCC={stage_summary['min_pcc']:.7f}, " f"NL2={stage_summary['max_nl2']:.7f}"
                    )
            if artifact_path:
                with Path(artifact_path).open("w") as artifact_file:
                    json.dump(all_results, artifact_file, indent=2)
            logger.info(f"attention real-token metrics {dtype_name}: {case}")

    assert not failures, "real-token composition gate failures: " + "; ".join(failures)


def _assert_supported_fp32_debug_tensors(debug, cache, *, slot, layer, start, end):
    expected_k, expected_v = _read_cache_prefix(cache, slot * NUM_LAYERS + layer, debug["logical_n"])
    owned = _owned_positions(start)
    natural_shards = {
        name: [ttnn.to_torch(shard).float() for shard in ttnn.get_device_tensors(debug[name])]
        for name in ("natural_k", "natural_v")
    }
    mask_shards = [ttnn.to_torch(shard).float() for shard in ttnn.get_device_tensors(debug["mask"])]
    valid_shards = [ttnn.to_torch(shard).float() for shard in ttnn.get_device_tensors(debug["query_valid"])]
    keys = torch.arange(debug["logical_n"])
    for device_idx in range(SP * TP):
        sp_coord, tp_coord = divmod(device_idx, TP)
        assert torch.equal(natural_shards["natural_k"][device_idx], expected_k[:, tp_coord : tp_coord + 1])
        assert torch.equal(natural_shards["natural_v"][device_idx], expected_v[:, tp_coord : tp_coord + 1])
        positions = torch.tensor(owned[sp_coord])
        valid = positions < end
        safe_positions = torch.where(valid, positions, 0)
        allowed = keys[None, :] <= safe_positions[:, None]
        actual_mask = mask_shards[device_idx][0, 0]
        assert torch.equal(actual_mask[allowed], torch.zeros_like(actual_mask[allowed]))
        assert torch.isneginf(actual_mask[~allowed]).all()
        actual_valid = valid_shards[device_idx][0, 0, :, 0]
        assert torch.equal(actual_valid, valid.to(actual_valid.dtype))


# The positive-pulse oracle is well conditioned, and the omitted/shifted host mutations prove the
# unchanged per-chip gate can detect a causal-boundary error.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(MESH_SHAPE, id="galaxy-4x8")], indirect=True)
def test_zero_score_positive_pulse_prefix_average(mesh_device):
    cases = (
        (0, 0, 0, 33, 0, 32),
        (1, 13, 224, 257, 1, 256),
        (0, 31, 992, 1025, 0, 1024),
        (1, 0, 2016, 2048, 1, 2047),
    )
    for _, _, start, end, prompt, pulse_position in cases:
        pulse_v = _positive_pulse_fixture("v", prompt, NUM_KV_HEADS, range(end))
        baseline_v = _positive_baseline_fixture("v", prompt, NUM_KV_HEADS, range(end))
        pulse_delta = pulse_v[:, :, pulse_position] - baseline_v[:, :, pulse_position]
        without_current = pulse_v.clone()
        without_current[:, :, pulse_position] = baseline_v[:, :, pulse_position]
        expected = _prefix_average_fixture(prompt, start, end, _positive_pulse_fixture)
        shifted_earlier = without_current.clone()
        shifted_earlier[:, :, pulse_position - 1] = (
            (baseline_v[:, :, pulse_position - 1] + pulse_delta).to(torch.bfloat16).float()
        )

        for mutation_name, mutated_v in (("omitted", without_current), ("shifted", shifted_earlier)):
            prefix_sum = torch.cumsum(mutated_v, dim=2)
            denominator = torch.arange(1, end + 1, dtype=torch.float32)[None, None, :, None]
            mutated = prefix_sum / denominator
            mutated = mutated.repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, dim=1)[:, :, start:end]
            mutation_metrics = {}
            owned = _owned_positions(start)
            exposed_position = pulse_position if mutation_name == "omitted" else pulse_position - 1
            exposed_sp = (exposed_position % (SP * LOCAL_SEQUENCE)) // LOCAL_SEQUENCE
            expected_rows = [position - start for position in owned[exposed_sp] if position < end]
            for tp_coord in range(TP):
                h0 = tp_coord * LOCAL_Q_HEADS
                wanted = expected[:, h0 : h0 + LOCAL_Q_HEADS, expected_rows]
                alternative = mutated[:, h0 : h0 + LOCAL_Q_HEADS, expected_rows]
                mutation_metrics[exposed_sp * TP + tp_coord] = _metrics(wanted, alternative)
            assert len(mutation_metrics) == TP and all(
                pcc < 0.999 or nl2 > 0.03 for pcc, nl2 in mutation_metrics.values()
            ), f"{mutation_name} causal pulse at {pulse_position} did not fail every exposed TP shard"

    mesh_config = MeshConfig(MESH_SHAPE, TP)
    for cache_dtype in (ttnn.bfloat16, ttnn.bfloat8_b):
        attention = FullCausalAttention(mesh_device, mesh_config, cache_dtype=cache_dtype)
        cache = allocate_kv_cache(mesh_device, mesh_config, cache_dtype=cache_dtype)
        _assert_attention_readiness(mesh_device, attention, cache, cache_dtype)
        for slot, layer, start, end, prompt, pulse_position in cases:
            actual_by_device = _run_attention_case(
                mesh_device,
                attention,
                cache,
                slot=slot,
                layer=layer,
                start=start,
                end=end,
                prompt=prompt,
                cache_dtype=cache_dtype,
                label=f"positive-pulse-{pulse_position}-{cache_dtype}",
                zero_qk=True,
                fixture_fn=_positive_pulse_fixture,
                enforce_metrics=False,
            )
            source_expected = _prefix_average_fixture(prompt, start, end, _positive_pulse_fixture)
            _, cache_v = _read_cache_prefix(cache, slot * NUM_LAYERS + layer, end)
            denominator = torch.arange(1, end + 1, dtype=torch.float32)[None, None, :, None]
            cache_expected = (torch.cumsum(cache_v, dim=2) / denominator).repeat_interleave(
                NUM_Q_HEADS // NUM_KV_HEADS, dim=1
            )[:, :, start:end]
            owned = _owned_positions(start)
            pcc_limit, nl2_limit = (0.999, 0.02) if cache_dtype == ttnn.bfloat16 else (0.999, 0.03)
            case_metrics = {}
            for device_idx, actual in actual_by_device.items():
                sp_coord, tp_coord = divmod(device_idx, TP)
                expected_rows = [position - start for position in owned[sp_coord] if position < end]
                # Preserve empty tensors for cross-call identity checks, but PCC/NL2 are undefined on them.
                if not expected_rows:
                    assert actual.numel() == 0
                    continue
                h0 = tp_coord * LOCAL_Q_HEADS
                source_metrics = _metrics(source_expected[:, h0 : h0 + LOCAL_Q_HEADS, expected_rows], actual)
                cache_metrics = _metrics(cache_expected[:, h0 : h0 + LOCAL_Q_HEADS, expected_rows], actual)
                case_metrics[str(device_idx)] = {"source": source_metrics, "cache": cache_metrics}
            logger.info(
                f"positive pulse source/cache metrics pulse={pulse_position}, dtype={cache_dtype}: {case_metrics}"
            )
            for device_idx, metrics_by_oracle in case_metrics.items():
                for oracle, (pcc, nl2) in metrics_by_oracle.items():
                    assert pcc >= pcc_limit, (
                        f"positive-pulse-{pulse_position}-{cache_dtype}/{oracle} chip={device_idx}: "
                        f"PCC={pcc:.7f}, NL2={nl2:.7f}"
                    )
                    assert nl2 <= nl2_limit, (
                        f"positive-pulse-{pulse_position}-{cache_dtype}/{oracle} chip={device_idx}: "
                        f"PCC={pcc:.7f}, NL2={nl2:.7f}"
                    )


# Keep semantic invariants independent of numerical reference gates so a precision failure cannot
# prevent cache immutability, causality, allocation refresh, validation, and reuse checks from running.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(MESH_SHAPE, id="galaxy-4x8")], indirect=True)
def test_full_causal_attention_structural_invariants(mesh_device, expect_error):
    mesh_config = MeshConfig(MESH_SHAPE, TP)
    mesh_device.enable_program_cache()
    for cache_dtype in (ttnn.bfloat16, ttnn.bfloat8_b):
        attention = FullCausalAttention(mesh_device, mesh_config, cache_dtype=cache_dtype)
        cache = allocate_kv_cache(mesh_device, mesh_config, cache_dtype=cache_dtype)
        _assert_attention_readiness(mesh_device, attention, cache, cache_dtype)
        persistent_addresses = _persistent_attention_addresses(attention)

        _run_attention_case(
            mesh_device,
            attention,
            cache,
            slot=1,
            layer=31,
            start=2016,
            end=2048,
            prompt=0,
            cache_dtype=cache_dtype,
            label=f"structural-exact-zero-full-prefix-{cache_dtype}",
            zero_v=True,
        )
        zero_q = _run_attention_case(
            mesh_device,
            attention,
            cache,
            slot=0,
            layer=13,
            start=224,
            end=257,
            prompt=1,
            cache_dtype=cache_dtype,
            label=f"structural-zero-q-{cache_dtype}",
            zero_q=True,
            fixture_fn=_positive_pulse_fixture,
            enforce_metrics=False,
        )
        zero_qk = _run_attention_case(
            mesh_device,
            attention,
            cache,
            slot=0,
            layer=13,
            start=224,
            end=257,
            prompt=1,
            cache_dtype=cache_dtype,
            label=f"structural-zero-qk-{cache_dtype}",
            zero_qk=True,
            fixture_fn=_positive_pulse_fixture,
            enforce_metrics=False,
        )
        assert zero_q.keys() == zero_qk.keys()
        assert all(torch.equal(zero_q[index], zero_qk[index]) for index in zero_q)

        clean = _run_attention_case(
            mesh_device,
            attention,
            cache,
            slot=1,
            layer=0,
            start=224,
            end=257,
            prompt=0,
            cache_dtype=cache_dtype,
            label=f"structural-clean-tail-{cache_dtype}",
            enforce_metrics=False,
        )
        dirty_inputs = _write_prefix(
            mesh_device,
            cache,
            slot=1,
            layer=0,
            end=GLOBAL_CHUNK,
            prompt=0,
            dirty_after=257,
        )
        dirty_q = _to_q(mesh_device, _physical_fixture("q", 0, NUM_Q_HEADS, 224))
        dirty_output = attention(
            dirty_q,
            cache,
            slot_idx=1,
            layer_idx=0,
            actual_start=224,
            actual_end=257,
        )
        ttnn.synchronize_device(mesh_device)
        owned = _owned_positions(224)
        for sp_coord in range(SP):
            rows = [row for row, position in enumerate(owned[sp_coord]) if position < 257]
            if not rows:
                continue
            for tp_coord in range(TP):
                device_idx = sp_coord * TP + tp_coord
                actual = ttnn.to_torch(ttnn.get_device_tensors(dirty_output)[device_idx]).float()[
                    :, :LOCAL_Q_HEADS, rows, :HEAD_DIM
                ]
                assert torch.equal(actual, clean[device_idx])
        dirty_output.deallocate(True)
        dirty_q.deallocate(True)
        for tensor in dirty_inputs:
            tensor.deallocate(True)

        second_cache = allocate_kv_cache(mesh_device, mesh_config, cache_dtype=cache_dtype)
        assert set(_addresses(cache.k) + _addresses(cache.v)).isdisjoint(
            _addresses(second_cache.k) + _addresses(second_cache.v)
        )
        first = _run_attention_case(
            mesh_device,
            attention,
            cache,
            slot=0,
            layer=0,
            start=0,
            end=33,
            prompt=0,
            cache_dtype=cache_dtype,
            label=f"structural-allocation-a-{cache_dtype}",
            enforce_metrics=False,
        )
        poison_inputs = _write_prefix(
            mesh_device,
            cache,
            slot=1,
            layer=31,
            end=33,
            prompt=1,
        )
        selected_q = _to_q(mesh_device, _physical_fixture("q", 0, NUM_Q_HEADS, 0))
        selected_after_poison = attention(
            selected_q,
            cache,
            slot_idx=0,
            layer_idx=0,
            actual_start=0,
            actual_end=33,
        )
        ttnn.synchronize_device(mesh_device)
        owned_zero = _owned_positions(0)
        for sp_coord in range(SP):
            rows = [row for row, position in enumerate(owned_zero[sp_coord]) if position < 33]
            for tp_coord in range(TP):
                device_idx = sp_coord * TP + tp_coord
                actual = ttnn.to_torch(ttnn.get_device_tensors(selected_after_poison)[device_idx]).float()[
                    :, :LOCAL_Q_HEADS, rows, :HEAD_DIM
                ]
                assert torch.equal(actual, first[device_idx])
        selected_after_poison.deallocate(True)
        selected_q.deallocate(True)
        for tensor in poison_inputs:
            tensor.deallocate(True)
        second = _run_attention_case(
            mesh_device,
            attention,
            second_cache,
            slot=0,
            layer=0,
            start=0,
            end=33,
            prompt=1,
            cache_dtype=cache_dtype,
            label=f"structural-allocation-b-{cache_dtype}",
            enforce_metrics=False,
        )
        returned = _run_attention_case(
            mesh_device,
            attention,
            cache,
            slot=0,
            layer=0,
            start=0,
            end=33,
            prompt=0,
            cache_dtype=cache_dtype,
            label=f"structural-allocation-a-return-{cache_dtype}",
            enforce_metrics=False,
        )
        assert any(not torch.equal(first[index], second[index]) for index in first)
        assert all(torch.equal(first[index], returned[index]) for index in first)

        # Keep both Q allocations live while replaying A->B->A with the same compiled signature;
        # this proves cached programs refresh both Q and cache addresses rather than reusing stale ones.
        q_a = _to_q(mesh_device, _physical_fixture("q", 0, NUM_Q_HEADS, 0))
        q_b = _to_q(mesh_device, _physical_fixture("q", 1, NUM_Q_HEADS, 0))
        assert all(a != b for a, b in zip(_addresses(q_a), _addresses(q_b)))
        direct_a = attention(q_a, cache, slot_idx=0, layer_idx=0, actual_start=0, actual_end=33)
        direct_b = attention(q_b, second_cache, slot_idx=0, layer_idx=0, actual_start=0, actual_end=33)
        direct_a_return = attention(q_a, cache, slot_idx=0, layer_idx=0, actual_start=0, actual_end=33)
        ttnn.synchronize_device(mesh_device)
        owned_direct = _owned_positions(0)
        for device_idx, (a_shard, b_shard, returned_shard) in enumerate(
            zip(
                ttnn.get_device_tensors(direct_a),
                ttnn.get_device_tensors(direct_b),
                ttnn.get_device_tensors(direct_a_return),
            )
        ):
            sp_coord = device_idx // TP
            valid_rows = [row for row, position in enumerate(owned_direct[sp_coord]) if position < 33]
            actual_a = ttnn.to_torch(a_shard).float()[:, :LOCAL_Q_HEADS, valid_rows, :HEAD_DIM]
            actual_b = ttnn.to_torch(b_shard).float()[:, :LOCAL_Q_HEADS, valid_rows, :HEAD_DIM]
            actual_return = ttnn.to_torch(returned_shard).float()[:, :LOCAL_Q_HEADS, valid_rows, :HEAD_DIM]
            assert torch.equal(actual_a, first[device_idx])
            assert torch.equal(actual_b, second[device_idx])
            assert torch.equal(actual_return, first[device_idx])
        for tensor in (direct_a, direct_b, direct_a_return, q_a, q_b):
            tensor.deallocate(True)

        stable_program_count = mesh_device.num_program_cache_entries()
        dram_before_replay = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank
        _run_attention_case(
            mesh_device,
            attention,
            cache,
            slot=0,
            layer=0,
            start=0,
            end=33,
            prompt=0,
            cache_dtype=cache_dtype,
            label=f"structural-warm-return-{cache_dtype}",
            enforce_metrics=False,
        )
        assert mesh_device.num_program_cache_entries() == stable_program_count
        dram_after_replay = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank
        assert dram_after_replay == dram_before_replay
        assert _persistent_attention_addresses(attention) == persistent_addresses

        request_cache_before = _cache_plane_snapshots(cache, 0)
        attention.validate_request(cache, slot_idx=0, layer_idx=0, actual_start=0, actual_end=1)
        _assert_cache_plane_unchanged(cache, 0, request_cache_before)
        with expect_error(TypeError, "eager Python int"):
            attention.validate_request(cache, slot_idx=True, layer_idx=0, actual_start=0, actual_end=1)
        with expect_error(ValueError, "tile-aligned"):
            attention.validate_request(cache, slot_idx=0, layer_idx=0, actual_start=1, actual_end=33)
        with expect_error(ValueError, "layer_idx"):
            attention.validate_request(cache, slot_idx=0, layer_idx=NUM_LAYERS, actual_start=0, actual_end=1)
        with expect_error(ValueError, "actual range"):
            attention.validate_request(
                cache,
                slot_idx=0,
                layer_idx=0,
                actual_start=2016,
                actual_end=MAX_SEQ_LEN + 1,
            )
        with expect_error(ValueError, "at most"):
            attention.validate_request(
                cache,
                slot_idx=0,
                layer_idx=0,
                actual_start=0,
                actual_end=GLOBAL_CHUNK + 1,
            )
        other_dtype = ttnn.bfloat8_b if cache_dtype == ttnn.bfloat16 else ttnn.bfloat16
        mismatched_cache = allocate_kv_cache(mesh_device, mesh_config, cache_dtype=other_dtype)
        mismatched_before = _cache_plane_snapshots(mismatched_cache, 0)
        with expect_error(ValueError, "match constructor"):
            attention.validate_request(
                mismatched_cache,
                slot_idx=0,
                layer_idx=0,
                actual_start=0,
                actual_end=1,
            )
        _assert_cache_plane_unchanged(cache, 0, request_cache_before)
        _assert_cache_plane_unchanged(mismatched_cache, 0, mismatched_before)

        valid_q = _to_q(mesh_device, _physical_fixture("q", 0, NUM_Q_HEADS, 0))
        valid_q_before = [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(valid_q)]
        valid_cache_before = _cache_plane_snapshots(cache, 0)
        with expect_error(ValueError, "tile-aligned"):
            attention(valid_q, cache, slot_idx=0, layer_idx=0, actual_start=1, actual_end=33)
        with expect_error(ValueError, "start < end"):
            attention(valid_q, cache, slot_idx=0, layer_idx=0, actual_start=32, actual_end=32)
        with expect_error(ValueError, "slot_idx"):
            attention(valid_q, cache, slot_idx=2, layer_idx=0, actual_start=0, actual_end=1)
        with expect_error(ValueError, "layer_idx"):
            attention(valid_q, cache, slot_idx=0, layer_idx=NUM_LAYERS, actual_start=0, actual_end=1)
        with expect_error(ValueError, "actual range"):
            attention(
                valid_q,
                cache,
                slot_idx=0,
                layer_idx=0,
                actual_start=2016,
                actual_end=MAX_SEQ_LEN + 1,
            )
        wrong_shape = _to_q(mesh_device, _physical_fixture("q", 0, NUM_Q_HEADS, 0)[:, :, :512])
        with expect_error(ValueError, "local shape"):
            attention(wrong_shape, cache, slot_idx=0, layer_idx=0, actual_start=0, actual_end=1)
        wrong_shape.deallocate(True)
        wrong_dtype = _to_q(
            mesh_device,
            _physical_fixture("q", 0, NUM_Q_HEADS, 0),
            dtype=ttnn.bfloat8_b,
        )
        with expect_error(ValueError, "attention Q must be"):
            attention(wrong_dtype, cache, slot_idx=0, layer_idx=0, actual_start=0, actual_end=1)
        wrong_dtype.deallocate(True)
        wrong_layout = _to_q(
            mesh_device,
            _physical_fixture("q", 0, NUM_Q_HEADS, 0),
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        with expect_error(ValueError, "TILE_LAYOUT"):
            attention(wrong_layout, cache, slot_idx=0, layer_idx=0, actual_start=0, actual_end=1)
        wrong_layout.deallocate(True)
        with expect_error(ValueError, "match constructor"):
            attention(valid_q, mismatched_cache, slot_idx=0, layer_idx=0, actual_start=0, actual_end=1)
        for shard, snapshot in zip(ttnn.get_device_tensors(valid_q), valid_q_before):
            assert torch.equal(ttnn.to_torch(shard), snapshot)
        _assert_cache_plane_unchanged(cache, 0, valid_cache_before)
        _assert_cache_plane_unchanged(mismatched_cache, 0, mismatched_before)
        assert _persistent_attention_addresses(attention) == persistent_addresses
        valid_q.deallocate(True)
        _run_attention_case(
            mesh_device,
            attention,
            cache,
            slot=0,
            layer=0,
            start=0,
            end=1,
            prompt=0,
            cache_dtype=cache_dtype,
            label=f"structural-valid-after-errors-{cache_dtype}",
            enforce_metrics=False,
        )


# Prove the production gather/reorder and absolute-position mask exactly at tile, SP, slab, and
# context-end boundaries in both supported cache dtypes.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(MESH_SHAPE, id="galaxy-4x8")], indirect=True)
def test_full_causal_attention_exact_boundaries(mesh_device):
    mesh_config = MeshConfig(MESH_SHAPE, TP)
    requested = {(0, 1), (0, 33), (224, 257), (1024, 1537), (1056, 2048), (2016, 2048)}
    scenarios = [case for case in RAW_SCENARIOS if (case[2], case[3]) in requested]
    assert {(case[2], case[3]) for case in scenarios} == requested
    # Run both sides and the exact edge of the production 512-key chunk with independent planes.
    scenarios.extend(
        (
            (0, 5, 480, 511, 0),
            (1, 17, 480, 512, 1),
            (0, 29, 480, 513, 2),
        )
    )

    for cache_dtype in (ttnn.bfloat16, ttnn.bfloat8_b):
        attention = FullCausalAttention(mesh_device, mesh_config, cache_dtype=cache_dtype)
        cache = allocate_kv_cache(mesh_device, mesh_config, cache_dtype=cache_dtype)
        _assert_attention_readiness(mesh_device, attention, cache, cache_dtype)
        for slot, layer, start, end, prompt in scenarios:
            attention.validate_request(
                cache,
                slot_idx=slot,
                layer_idx=layer,
                actual_start=start,
                actual_end=end,
            )
            prefix_inputs = _write_prefix(
                mesh_device,
                cache,
                slot=slot,
                layer=layer,
                end=end,
                prompt=prompt,
            )
            logical_n = math.ceil(end / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
            batch_index = slot * NUM_LAYERS + layer
            natural_k = attention._gather_and_reorder(
                cache.k,
                attention.gathered_k,
                batch_index=batch_index,
                logical_n=logical_n,
            )
            natural_v = attention._gather_and_reorder(
                cache.v,
                attention.gathered_v,
                batch_index=batch_index,
                logical_n=logical_n,
            )
            mask, query_valid = attention._build_mask(
                actual_start=start,
                actual_end=end,
                logical_n=logical_n,
            )
            attention._require_sdpa_l1()
            debug = {
                "natural_k": natural_k,
                "natural_v": natural_v,
                "mask": mask,
                "query_valid": query_valid,
                "logical_n": logical_n,
            }
            ttnn.synchronize_device(mesh_device)
            _assert_supported_fp32_debug_tensors(
                debug,
                cache,
                slot=slot,
                layer=layer,
                start=start,
                end=end,
            )
            for tensor in (natural_k, natural_v, mask, query_valid):
                tensor.deallocate(True)

            # Exercise the full production call after the private-coordinate proof. Every padded
            # query row is an exact zero, including SP ranks with no valid query for short intervals.
            q = _to_q(mesh_device, _physical_fixture("q", prompt, NUM_Q_HEADS, start))
            output = attention(
                q,
                cache,
                slot_idx=slot,
                layer_idx=layer,
                actual_start=start,
                actual_end=end,
            )
            ttnn.synchronize_device(mesh_device)
            owned = _owned_positions(start)
            for device_idx, shard in enumerate(ttnn.get_device_tensors(output)):
                actual = ttnn.to_torch(shard).float()[:, :LOCAL_Q_HEADS, :, :HEAD_DIM]
                assert torch.isfinite(actual).all()
                sp_coord = device_idx // TP
                valid = torch.tensor([position < end for position in owned[sp_coord]])
                assert torch.count_nonzero(actual[:, :, ~valid, :]) == 0
            output.deallocate(True)
            q.deallocate(True)
            for tensor in prefix_inputs:
                tensor.deallocate(True)


# Attend over a cache holding more than the default two sequences. The gather selects a plane by
# batch = slot * NUM_LAYERS + layer, so a slot above the old fixed pair is the only thing that
# exercises that arithmetic past its first two values. A wrong stride there would return a
# NEIGHBOURING slot's KV rather than fail, so every slot is filled with a different prompt before any
# of them is read, and each is graded against its own reference.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(MESH_SHAPE, id="galaxy-4x8")], indirect=True)
def test_full_causal_attention_reads_extra_user_slots(mesh_device, expect_error):
    mesh_config = MeshConfig(MESH_SHAPE, TP)
    slots, cache_dtype = 4, ttnn.bfloat16
    # Layers chosen so the batch index is not a multiple of the layer count for any slot.
    cases = [(0, 0), (1, 13), (2, 31), (3, 7)]
    attention = FullCausalAttention(mesh_device, mesh_config, cache_dtype=cache_dtype, num_users=slots)
    cache = allocate_kv_cache(mesh_device, mesh_config, cache_dtype=cache_dtype, num_users=slots)
    assert (attention.num_users, cache.num_users) == (slots, slots)
    _assert_attention_readiness(mesh_device, attention, cache, cache_dtype)

    start, end = 224, 257
    prefix_inputs = []
    for slot, layer in cases:
        prefix_inputs.extend(_write_prefix(mesh_device, cache, slot=slot, layer=layer, end=end, prompt=slot))
    ttnn.synchronize_device(mesh_device)

    per_slot = {}
    for slot, layer in cases:
        per_slot[slot] = _run_attention_case(
            mesh_device,
            attention,
            cache,
            slot=slot,
            layer=layer,
            start=start,
            end=end,
            prompt=slot,
            cache_dtype=cache_dtype,
            label=f"extra-slot-{slot}-layer-{layer}",
        )
    # Distinct prompts must give distinct outputs; equality would mean two slots read one plane.
    # SP ranks owning no position below `end` return no rows for this range, and two empty tensors
    # compare equal, so those chips carry no signal and are skipped rather than counted as matches.
    for slot in range(1, slots):
        compared = [
            device_idx
            for device_idx, output in per_slot[slot].items()
            if output.numel() and not torch.equal(output, per_slot[0][device_idx])
        ]
        populated = [index for index, output in per_slot[slot].items() if output.numel()]
        assert populated, f"slot {slot} produced no rows to compare"
        assert compared == populated, f"slot {slot} matched slot 0 on chips {sorted(set(populated) - set(compared))}"

    valid_q = _to_q(mesh_device, _physical_fixture("q", 0, NUM_Q_HEADS, start))
    with expect_error(ValueError, f"slot_idx {slots} out of range"):
        attention(valid_q, cache, slot_idx=slots, layer_idx=0, actual_start=start, actual_end=end)
    with expect_error(ValueError, f"slot_idx {slots} out of range"):
        attention.validate_request(cache, slot_idx=slots, layer_idx=0, actual_start=start, actual_end=end)
    # An attention built for the default pair must refuse this cache instead of addressing 2 of its 4
    # slots: the two disagree about the packed batch extent the gather indexes into.
    default_attention = FullCausalAttention(mesh_device, mesh_config, cache_dtype=cache_dtype)
    with expect_error(ValueError, "cache metadata must be"):
        default_attention.validate_request(cache, slot_idx=0, layer_idx=0, actual_start=start, actual_end=end)

    valid_q.deallocate(True)
    for tensor in prefix_inputs:
        tensor.deallocate(True)
    cache.k.deallocate(True)
    cache.v.deallocate(True)


# The periodic raw fixture is retained as precision characterization against its independent source
# oracle. This separate functional gate compares production only with stock full-causal FP32 SDPA on
# the exact same BF16 Q and exact stored-cache K/V, using limits selected before the run.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(MESH_SHAPE, id="galaxy-4x8")], indirect=True)
def test_full_causal_attention_matches_stock_causal_on_raw_scenarios(mesh_device, tmp_path):
    artifact_dir = Path(os.environ.get("LLAMA_PREFILL_ATTENTION_PARITY_DIR", tmp_path))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    mesh_config = MeshConfig(MESH_SHAPE, TP)
    all_results = {}
    failures = []

    for cache_dtype in (ttnn.bfloat16, ttnn.bfloat8_b):
        dtype_name = "bfloat16" if cache_dtype == ttnn.bfloat16 else "bfloat8_b"
        attention = FullCausalAttention(mesh_device, mesh_config, cache_dtype=cache_dtype)
        cache = allocate_kv_cache(mesh_device, mesh_config, cache_dtype=cache_dtype)
        _assert_attention_readiness(mesh_device, attention, cache, cache_dtype)
        dtype_results = []
        for slot, layer, start, end, prompt in RAW_SCENARIOS:
            attention.validate_request(
                cache,
                slot_idx=slot,
                layer_idx=layer,
                actual_start=start,
                actual_end=end,
            )
            prefix_inputs = _write_prefix(
                mesh_device,
                cache,
                slot=slot,
                layer=layer,
                end=end,
                prompt=prompt,
            )
            local_q = _to_q(mesh_device, _physical_fixture("q", prompt, NUM_Q_HEADS, start))
            production = attention(
                local_q,
                cache,
                slot_idx=slot,
                layer_idx=layer,
                actual_start=start,
                actual_end=end,
            )
            logical_n = math.ceil(end / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
            batch_index = slot * NUM_LAYERS + layer
            natural_k = attention._gather_and_reorder(
                cache.k,
                attention.gathered_k,
                batch_index=batch_index,
                logical_n=logical_n,
            )
            natural_v = attention._gather_and_reorder(
                cache.v,
                attention.gathered_v,
                batch_index=batch_index,
                logical_n=logical_n,
            )
            full_q_host = _fixture("q", prompt, NUM_Q_HEADS, range(logical_n))
            full_q = ttnn.from_torch(
                full_q_host.to(torch.bfloat16),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 1)),
            )
            stock = ttnn.transformer.scaled_dot_product_attention(
                full_q,
                natural_k,
                natural_v,
                is_causal=True,
                scale=HEAD_DIM**-0.5,
                program_config=attention.program_config,
                compute_kernel_config=attention.compute_kernel_config,
            )
            ttnn.synchronize_device(mesh_device)

            owned = _owned_positions(start)
            production_shards = [ttnn.to_torch(shard).float() for shard in ttnn.get_device_tensors(production)]
            stock_shards = [ttnn.to_torch(shard).float() for shard in ttnn.get_device_tensors(stock)]
            per_chip = {}
            for sp_coord in range(SP):
                local_rows = [row for row, position in enumerate(owned[sp_coord]) if position < end]
                positions = [position for position in owned[sp_coord] if position < end]
                if not local_rows:
                    continue
                for tp_coord in range(TP):
                    device_idx = sp_coord * TP + tp_coord
                    actual = production_shards[device_idx][:, :LOCAL_Q_HEADS, local_rows, :HEAD_DIM]
                    expected = stock_shards[device_idx][:, :LOCAL_Q_HEADS, positions, :HEAD_DIM]
                    assert torch.isfinite(actual).all()
                    assert torch.isfinite(expected).all()
                    metrics = _diagnostic_metrics(expected, actual)
                    assert all(math.isfinite(value) for value in metrics.values())
                    per_chip[str(device_idx)] = metrics
                    if metrics["pcc"] < 0.9999 or metrics["nl2"] > 0.01:
                        failures.append(
                            f"{dtype_name}-s{slot}-l{layer}-{start}-{end}-p{prompt} chip={device_idx}: "
                            f"PCC={metrics['pcc']:.7f}, NL2={metrics['nl2']:.7f}"
                        )
            for tp_coord in range(TP):
                replica = stock_shards[tp_coord][:, :LOCAL_Q_HEADS, :logical_n, :HEAD_DIM]
                for sp_coord in range(1, SP):
                    candidate = stock_shards[sp_coord * TP + tp_coord][:, :LOCAL_Q_HEADS, :logical_n, :HEAD_DIM]
                    assert torch.equal(candidate, replica)
            case = {
                "slot": slot,
                "layer": layer,
                "start": start,
                "end": end,
                "prompt": prompt,
                "min_pcc": min(value["pcc"] for value in per_chip.values()),
                "max_nl2": max(value["nl2"] for value in per_chip.values()),
                "per_chip": per_chip,
            }
            dtype_results.append(case)
            all_results[dtype_name] = dtype_results
            with (artifact_dir / "stock-parity.json").open("w") as artifact_file:
                json.dump(all_results, artifact_file, indent=2)
            for tensor in (production, local_q, stock, full_q, natural_k, natural_v):
                tensor.deallocate(True)
            for tensor in prefix_inputs:
                tensor.deallocate(True)

    assert not failures, "production-vs-stock raw parity failures: " + "; ".join(failures)
