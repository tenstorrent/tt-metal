# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer, Phi3RotaryEmbedding

import ttnn
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.functional_decoder import DEFAULT_PAGE_SIZE, FunctionalDecoder
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import comp_pcc

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
LAYER_IDX = 0
REAL_WEIGHT_SHARD = Path(
    "/huggingface/hub/models--microsoft--Phi-3.5-mini-instruct/"
    "snapshots/2fe192450127e6a83f7441aef6e3ca586c338b77/model-00001-of-00002.safetensors"
)


def _config():
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    config._attn_implementation = "eager"
    return config


def _key(suffix):
    return f"model.layers.{LAYER_IDX}.{suffix}"


def _synthetic_state(config, seed=20260728):
    generator = torch.Generator().manual_seed(seed)
    hidden = config.hidden_size
    inter = config.intermediate_size

    # Deterministic fixture derived from layer-0 real-weight statistics printed
    # by test_real_weight_paged_prefill_and_decode.
    def sample(shape, mean, std):
        return (torch.randn(*shape, generator=generator) * std + mean).to(torch.bfloat16)

    return {
        _key("input_layernorm.weight"): sample((hidden,), 0.00829245, 0.02295496),
        _key("post_attention_layernorm.weight"): sample((hidden,), 0.03923744, 0.00945584),
        _key("self_attn.qkv_proj.weight"): sample((3 * hidden, hidden), 0.00000262, 0.02379715),
        _key("self_attn.o_proj.weight"): sample((hidden, hidden), -0.00000081, 0.01751270),
        _key("mlp.gate_up_proj.weight"): sample((2 * inter, hidden), -0.00001401, 0.03248470),
        _key("mlp.down_proj.weight"): sample((hidden, inter), 0.00000275, 0.03603584),
    }


def _zero_state(config):
    hidden = config.hidden_size
    inter = config.intermediate_size
    return {
        _key("input_layernorm.weight"): torch.ones(hidden, dtype=torch.bfloat16),
        _key("post_attention_layernorm.weight"): torch.ones(hidden, dtype=torch.bfloat16),
        _key("self_attn.qkv_proj.weight"): torch.zeros(3 * hidden, hidden, dtype=torch.bfloat16),
        _key("self_attn.o_proj.weight"): torch.zeros(hidden, hidden, dtype=torch.bfloat16),
        _key("mlp.gate_up_proj.weight"): torch.zeros(2 * inter, hidden, dtype=torch.bfloat16),
        _key("mlp.down_proj.weight"): torch.zeros(hidden, inter, dtype=torch.bfloat16),
    }


def _real_state():
    if not REAL_WEIGHT_SHARD.is_file():
        pytest.skip(f"real Phi-3.5 weight shard not found at {REAL_WEIGHT_SHARD}")
    prefix = f"model.layers.{LAYER_IDX}."
    with safe_open(REAL_WEIGHT_SHARD, framework="pt", device="cpu") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys() if key.startswith(prefix)}


def _rms_norm(x, weight, eps):
    return weight * x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)


def _rope(config, positions, *, use_long):
    head_dim = config.hidden_size // config.num_attention_heads
    factors = config.rope_scaling["long_factor" if use_long else "short_factor"]
    exponent = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    inv = 1.0 / (torch.tensor(factors) * float(config.rope_theta) ** exponent)
    emb = torch.cat((positions.float().unsqueeze(1) * inv.unsqueeze(0),) * 2, dim=-1)
    amplitude = math.sqrt(
        1
        + math.log(config.max_position_embeddings / config.original_max_position_embeddings)
        / math.log(config.original_max_position_embeddings)
    )
    return emb.cos() * amplitude, emb.sin() * amplitude


def _rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _project_qkv(config, state, hidden, positions, *, use_long):
    normalized = _rms_norm(hidden, state[_key("input_layernorm.weight")], config.rms_norm_eps).to(torch.bfloat16)
    qkv = F.linear(normalized, state[_key("self_attn.qkv_proj.weight")])
    q, k, v = qkv.chunk(3, dim=-1)
    batch, seq, _ = q.shape
    head_dim = config.hidden_size // config.num_attention_heads
    q = q.view(batch, seq, config.num_attention_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq, config.num_key_value_heads, head_dim).transpose(1, 2)
    v = v.view(batch, seq, config.num_key_value_heads, head_dim).transpose(1, 2)
    cos, sin = _rope(config, positions, use_long=use_long)
    cos, sin = cos[None, None], sin[None, None]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin, v


def _finish(config, state, residual, attended):
    joined = attended.transpose(1, 2).reshape(residual.shape)
    post_attention = residual + F.linear(joined, state[_key("self_attn.o_proj.weight")])
    normalized = _rms_norm(post_attention, state[_key("post_attention_layernorm.weight")], config.rms_norm_eps).to(
        torch.bfloat16
    )
    gate, up = F.linear(normalized, state[_key("mlp.gate_up_proj.weight")]).chunk(2, dim=-1)
    return post_attention + F.linear(F.silu(gate) * up, state[_key("mlp.down_proj.weight")])


def _reference_prefill(config, state, hidden):
    positions = torch.arange(hidden.shape[1])
    q, k, v = _project_qkv(
        config, state, hidden, positions, use_long=hidden.shape[1] > config.original_max_position_embeddings
    )
    scores = q.float() @ k.float().transpose(-2, -1) / math.sqrt(q.shape[-1])
    causal = torch.ones(hidden.shape[1], hidden.shape[1], dtype=torch.bool).tril()
    scores.masked_fill_(~causal, torch.finfo(scores.dtype).min)
    attended = torch.softmax(scores, dim=-1).to(v.dtype) @ v
    return _finish(config, state, hidden, attended), (k, v)


def _reference_prefill_last_token(config, state, hidden):
    positions = torch.arange(hidden.shape[1])
    q, k, v = _project_qkv(config, state, hidden, positions, use_long=True)
    q_last = q[:, :, -1:, :]
    scores = q_last.float() @ k.float().transpose(-2, -1) / math.sqrt(q.shape[-1])
    attended = torch.softmax(scores, dim=-1).to(v.dtype) @ v
    return _finish(config, state, hidden[:, -1:, :], attended)


def _reference_decode(config, state, hidden, position, past):
    q, k, v = _project_qkv(
        config,
        state,
        hidden,
        torch.tensor([position]),
        use_long=position + 1 > config.original_max_position_embeddings,
    )
    keys = torch.cat((past[0], k), dim=-2)
    values = torch.cat((past[1], v), dim=-2)
    attended = torch.softmax(q.float() @ keys.float().transpose(-2, -1) / math.sqrt(q.shape[-1]), dim=-1)
    attended = attended.to(values.dtype) @ values
    return _finish(config, state, hidden, attended)


def _reference_decode_zero_prefix(config, state, hidden, position, *, use_long=True):
    """Reference a full-context cache whose prefix K/V rows are exactly zero."""
    positions = torch.as_tensor(position).reshape(-1)
    if positions.numel() > 1:
        return torch.cat(
            [
                _reference_decode_zero_prefix(
                    config,
                    state,
                    hidden[index : index + 1],
                    int(item),
                    use_long=use_long,
                )
                for index, item in enumerate(positions)
            ],
            dim=0,
        )
    q, k, v = _project_qkv(config, state, hidden, positions, use_long=use_long)
    last_score = (q.float() * k.float()).sum(-1) / math.sqrt(q.shape[-1])
    denominator = positions.reshape(-1, 1, 1) + torch.exp(last_score)
    last_probability = torch.exp(last_score) / denominator
    attended = last_probability.unsqueeze(-1) * v.float()
    return _finish(config, state, hidden, attended.to(v.dtype))


def _to_tt_prefill(hidden, mesh_device):
    return ttnn.from_torch(
        hidden.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _to_tt_decode(hidden, mesh_device):
    return ttnn.from_torch(
        hidden.transpose(0, 1).unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _to_torch_prefill(value):
    return ttnn.to_torch(ttnn.get_device_tensors(value)[0]).squeeze(0)


def _to_torch_decode(value):
    return ttnn.to_torch(ttnn.get_device_tensors(value)[0]).squeeze(0).transpose(0, 1)


def _page_table(batch, max_context, mesh_device, *, permute=False):
    blocks = math.ceil(max_context / DEFAULT_PAGE_SIZE)
    table = torch.arange(batch * blocks, dtype=torch.int32).reshape(batch, blocks)
    if permute:
        table = table.flip(-1)
    return ttnn.from_torch(
        table,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _positions(values, mesh_device):
    return ttnn.from_torch(
        torch.tensor(values, dtype=torch.int32),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _assert_pcc(label, reference, actual, threshold=0.995):
    passed, message = comp_pcc(reference.float(), actual.float(), threshold)
    print(f"PCC_RESULT path={label} threshold={threshold} {message}")
    assert passed, message


def _pairwise_diagnostics(label, left, right):
    left_float = left.float()
    right_float = right.float()
    difference = (left_float - right_float).abs()
    finite = bool(torch.isfinite(left_float).all() and torch.isfinite(right_float).all())
    pcc_passed, pcc_message = comp_pcc(left_float, right_float, 0.999999)
    metrics = {
        "equal": torch.equal(left, right),
        "mismatch_count": int(torch.count_nonzero(left != right)),
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "finite": finite,
        "pcc_passed_0.999999": pcc_passed,
    }
    print(f"TRACE_PAIR label={label} metrics={metrics} pcc={pcc_message}")
    return metrics


def test_static_contract_and_runtime_fallback_audit():
    assert issubclass(FunctionalDecoder, LightweightModule)
    runtime = (
        FunctionalDecoder._norm,
        FunctionalDecoder._mlp,
        FunctionalDecoder._prefill_rope,
        FunctionalDecoder._offset_causal_mask,
        FunctionalDecoder.prefill_forward,
        FunctionalDecoder._decode_rope,
        FunctionalDecoder.decode_forward,
        FunctionalDecoder.forward,
    )
    for method in runtime:
        source = inspect.getsource(method)
        for forbidden in ("torch", "from_torch", "to_torch", ".cpu(", "all_gather", "all_reduce"):
            assert forbidden not in source, (method.__name__, forbidden)


def test_longrope_tables_match_hf_rotary():
    config = _config()
    rotary = Phi3RotaryEmbedding(config)
    for use_long, values in ((False, [0, 4095]), (True, [0, 4095, 4096, 131071])):
        positions = torch.tensor([values])
        hidden = torch.zeros(1, positions.shape[1], config.hidden_size, dtype=torch.bfloat16)
        # Phi selects one factor set for the whole request based on its maximum
        # position; compare the short and long tables in separate requests.
        hf_cos, hf_sin = rotary(hidden, positions)
        for index, position in enumerate(values):
            expected_cos, expected_sin = _rope(config, torch.tensor([position]), use_long=use_long)
            torch.testing.assert_close(hf_cos[:, index].float(), expected_cos.float(), atol=0.01, rtol=0.01)
            torch.testing.assert_close(hf_sin[:, index].float(), expected_sin.float(), atol=0.01, rtol=0.01)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", [31, 32, 33, 63, 64, 65])
def test_paged_prefill_synthetic_matches_reference(mesh_device, seq_len):
    config = _config()
    state = _synthetic_state(config)
    max_context = math.ceil(seq_len / DEFAULT_PAGE_SIZE) * DEFAULT_PAGE_SIZE
    decoder = FunctionalDecoder.from_state_dict(
        state, hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device, max_context=max_context
    )
    hidden = torch.randn(1, seq_len, config.hidden_size, generator=torch.Generator().manual_seed(seq_len)).to(
        torch.bfloat16
    )
    reference, _ = _reference_prefill(config, state, hidden)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    actual = decoder.prefill_forward(
        _to_tt_prefill(hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=_page_table(1, max_context, mesh_device, permute=True),
    )
    _assert_pcc(f"prefill-{seq_len}", reference, _to_torch_prefill(actual))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_paged_prefill_batch2_cache_routing(mesh_device):
    config = _config()
    state = _synthetic_state(config)
    batch, seq_len, max_context = 2, 33, 64
    decoder = FunctionalDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_context=max_context,
    )
    hidden = torch.randn(batch, seq_len, config.hidden_size, generator=torch.Generator().manual_seed(233)).to(
        torch.bfloat16
    )
    reference, _ = _reference_prefill(config, state, hidden)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    actual = decoder.prefill_forward(
        _to_tt_prefill(hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=_page_table(batch, max_context, mesh_device, permute=True),
    )
    _assert_pcc("prefill-batch2-33", reference, _to_torch_prefill(actual))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", [131_071, 131_072])
def test_paged_prefill_long_non_aligned_context(mesh_device, seq_len):
    """Exercise the longest non-aligned and exact advertised lengths."""
    config = _config()
    state = _zero_state(config)
    decoder = FunctionalDecoder.from_state_dict(
        state, hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device, max_context=seq_len
    )
    hidden = torch.zeros(1, seq_len, config.hidden_size, dtype=torch.bfloat16)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    actual = decoder.prefill_forward(
        _to_tt_prefill(hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=_page_table(1, seq_len, mesh_device, permute=True),
    )
    result = _to_torch_prefill(actual)
    assert tuple(result.shape) == tuple(hidden.shape)
    assert torch.count_nonzero(result) == 0
    print(f"CONTEXT_RESULT mode=prefill batch=1 context={seq_len} chunked=true non_aligned=true")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_paged_prefill_nonzero_chunk_boundary_last_token_pcc(mesh_device):
    config = _config()
    state = _synthetic_state(config)
    seq_len = 32_769
    decoder = FunctionalDecoder.from_state_dict(
        state, hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device, max_context=seq_len
    )
    hidden = (torch.randn(1, seq_len, config.hidden_size, generator=torch.Generator().manual_seed(32769)) * 0.02).to(
        torch.bfloat16
    )
    reference = _reference_prefill_last_token(config, state, hidden)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    actual = decoder.prefill_forward(
        _to_tt_prefill(hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=_page_table(1, seq_len, mesh_device, permute=True),
    )
    _assert_pcc("prefill-nonzero-32769-last-token", reference, _to_torch_prefill(actual)[:, -1:, :])


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_paged_decode_synthetic_matches_reference(mesh_device):
    config = _config()
    state = _synthetic_state(config)
    decoder = FunctionalDecoder.from_state_dict(
        state, hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device, max_context=64
    )
    page_table = _page_table(1, 64, mesh_device, permute=True)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    prefill = torch.randn(1, 33, config.hidden_size, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    _, past = _reference_prefill(config, state, prefill)
    decoder.prefill_forward(
        _to_tt_prefill(prefill, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
    )
    hidden = torch.randn(1, 1, config.hidden_size, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)
    reference = _reference_decode(config, state, hidden, 33, past)
    actual = decoder.decode_forward(
        _to_tt_decode(hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=_positions([33], mesh_device),
        use_long_rope=False,
    )
    _assert_pcc("decode-33", reference, _to_torch_decode(actual))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_real_weight_paged_prefill_and_decode(mesh_device):
    config = _config()
    state = _real_state()
    decoder = FunctionalDecoder.from_state_dict(
        state, hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device, max_context=64
    )
    page_table = _page_table(1, 64, mesh_device, permute=True)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    generator = torch.Generator().manual_seed(3500)
    prefill = (torch.randn(1, 33, config.hidden_size, generator=generator) * 0.2).to(torch.bfloat16)
    prefill_reference, past = _reference_prefill(config, state, prefill)
    hf_layer = Phi3DecoderLayer(config, LAYER_IDX).eval()
    hf_layer.load_state_dict({key.split(f"model.layers.{LAYER_IDX}.", 1)[1]: value for key, value in state.items()})
    position_ids = torch.arange(prefill.shape[1]).unsqueeze(0)
    rotary = Phi3RotaryEmbedding(config)
    with torch.no_grad():
        hf_reference = hf_layer(
            prefill,
            position_ids=position_ids,
            position_embeddings=rotary(prefill, position_ids),
            use_cache=False,
        )
    _assert_pcc("manual-reference-vs-hf-layer", hf_reference, prefill_reference)
    for key, value in sorted(state.items()):
        stats = value.float()
        print(
            f"REAL_WEIGHT_STATS key={key} shape={tuple(value.shape)} "
            f"mean={stats.mean().item():.8f} std={stats.std().item():.8f} "
            f"min={stats.min().item():.8f} max={stats.max().item():.8f}"
        )
    prefill_actual = decoder.prefill_forward(
        _to_tt_prefill(prefill, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
    )
    _assert_pcc("real-prefill-33", prefill_reference, _to_torch_prefill(prefill_actual))
    hidden = (torch.randn(1, 1, config.hidden_size, generator=generator) * 0.2).to(torch.bfloat16)
    decode_reference = _reference_decode(config, state, hidden, 33, past)
    decode_actual = decoder.decode_forward(
        _to_tt_decode(hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=_positions([33], mesh_device),
        use_long_rope=False,
    )
    _assert_pcc("real-decode-33", decode_reference, _to_torch_decode(decode_actual))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_real_weight_decode_at_advertised_context(mesh_device):
    config = _config()
    state = _real_state()
    decoder = FunctionalDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        max_context=config.max_position_embeddings,
    )
    page_table = _page_table(1, config.max_position_embeddings, mesh_device, permute=True)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    position = config.max_position_embeddings - 1
    hidden = (torch.randn(1, 1, config.hidden_size, generator=torch.Generator().manual_seed(131072)) * 0.2).to(
        torch.bfloat16
    )
    reference = _reference_decode_zero_prefix(config, state, hidden, position)
    actual = decoder.decode_forward(
        _to_tt_decode(hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=_positions([position], mesh_device),
        use_long_rope=True,
    )
    _assert_pcc("real-decode-context-131072", reference, _to_torch_decode(actual))
    print("CONTEXT_RESULT mode=decode batch=1 context=131072 physical_limit=false")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
def test_decode_trace_replay_is_deterministic(mesh_device, batch):
    config = _config()
    state = _synthetic_state(config)
    decoder = FunctionalDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_context=64,
    )
    page_table = _page_table(batch, 64, mesh_device, permute=True)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    hidden = torch.randn(batch, 1, config.hidden_size, generator=torch.Generator().manual_seed(100 + batch)).to(
        torch.bfloat16
    )
    tt_hidden = _to_tt_decode(hidden, mesh_device)
    positions = [33] if batch == 1 else list(range(1, batch + 1))
    current_positions = _positions(positions, mesh_device)

    def decode():
        return decoder.decode_forward(
            tt_hidden,
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
            current_positions=current_positions,
            use_long_rope=False,
        )

    compile_output = decode()
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = decode()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    try:
        captured = ttnn.to_torch(ttnn.get_device_tensors(trace_output)[0]).clone()
        replayed = []
        for _ in range(3):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            replayed.append(ttnn.to_torch(ttnn.get_device_tensors(trace_output)[0]).clone())
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    trace_values = [captured, *replayed]
    trace_labels = ["capture", "R1", "R2", "R3"]
    trace_metrics = []
    replay_metrics = []
    for left_index in range(len(trace_values)):
        for right_index in range(left_index + 1, len(trace_values)):
            metric = _pairwise_diagnostics(
                f"batch{batch}:{trace_labels[left_index]}-{trace_labels[right_index]}",
                trace_values[left_index],
                trace_values[right_index],
            )
            trace_metrics.append(metric)
            if left_index > 0:
                replay_metrics.append(metric)

    eager_values = []
    for eager_index in range(3):
        eager_key_cache, eager_value_cache = decoder.create_paged_kv_cache()
        eager_output = decoder.decode_forward(
            tt_hidden,
            key_cache=eager_key_cache,
            value_cache=eager_value_cache,
            page_table=page_table,
            current_positions=current_positions,
            use_long_rope=False,
        )
        ttnn.synchronize_device(mesh_device)
        eager_values.append(ttnn.to_torch(ttnn.get_device_tensors(eager_output)[0]).clone())
    eager_metrics = []
    for left_index in range(len(eager_values)):
        for right_index in range(left_index + 1, len(eager_values)):
            eager_metrics.append(
                _pairwise_diagnostics(
                    f"batch{batch}:E{left_index + 1}-E{right_index + 1}",
                    eager_values[left_index],
                    eager_values[right_index],
                )
            )

    replay_eager_metric = _pairwise_diagnostics(f"batch{batch}:R1-E1", replayed[0], eager_values[0])
    reference = _reference_decode_zero_prefix(config, state, hidden, positions, use_long=False)
    replay_as_decode = replayed[0].squeeze(0).transpose(0, 1)
    _assert_pcc(f"trace-decode-reference-batch{batch}", reference, replay_as_decode)

    # Trace capture executes while recording and its returned tensor is not a
    # steady-state replay result. Determinism is therefore gated on replay
    # versus replay, with a fresh-cache eager result as the semantic control.
    assert all(metric["equal"] for metric in replay_metrics), replay_metrics
    assert all(metric["equal"] for metric in eager_metrics), eager_metrics
    assert replay_eager_metric["equal"], replay_eager_metric
    print(f"TRACE_RESULT batch={batch} replays=3 eager_runs=3 bitwise_deterministic=true")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_long_rope_decode_trace_matches_reference(mesh_device):
    config = _config()
    state = _synthetic_state(config)
    position = config.original_max_position_embeddings
    decoder = FunctionalDecoder.from_state_dict(
        state, hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device, max_context=position + 1
    )
    page_table = _page_table(1, position + 1, mesh_device, permute=True)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    hidden = torch.randn(1, 1, config.hidden_size, generator=torch.Generator().manual_seed(4096)).to(torch.bfloat16)
    tt_hidden = _to_tt_decode(hidden, mesh_device)
    current_positions = _positions([position], mesh_device)
    decoder.decode_forward(
        tt_hidden,
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=current_positions,
        use_long_rope=True,
    )
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = decoder.decode_forward(
        tt_hidden,
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=current_positions,
        use_long_rope=True,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        actual = _to_torch_decode(trace_output)
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    reference = _reference_decode_zero_prefix(config, state, hidden, position)
    _assert_pcc("trace-decode-long-rope-position-4096", reference, actual)
