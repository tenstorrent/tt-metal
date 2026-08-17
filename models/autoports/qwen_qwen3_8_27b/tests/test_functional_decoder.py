# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Target-shape correctness and runtime-contract tests for Qwen3.8-27B."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.functional_decoder import FunctionalDecoder, _internal_state_dict
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.demos.blackhole.qwen36.tt.rope import compute_rope_freqs
from models.experimental.gated_attention_gated_deltanet.torch_functional.gated_attention import gated_attention_forward
from models.experimental.gated_attention_gated_deltanet.torch_functional.gated_deltanet import gated_deltanet_forward

MODEL_ID = "Qwen/Qwen3.8-27B"
DEVICE_PARAMS = [{"l1_small_size": 24576, "num_command_queues": 2, "trace_region_size": 0}]
PCC_BAR = 0.995


def _checkpoint_root():
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            MODEL_ID,
            allow_patterns=["config.json", "model.safetensors.index.json", "model-*.safetensors"],
            local_files_only=True,
        )
    )


@pytest.fixture(scope="module")
def hf_config():
    return AutoConfig.from_pretrained(_checkpoint_root(), local_files_only=True)


def _load_real_layer(layer_idx):
    root = _checkpoint_root()
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer_idx}."
    keys = [key for key in index if key.startswith(prefix)]
    state = {}
    for filename in sorted({index[key] for key in keys}):
        with safe_open(root / filename, framework="pt", device="cpu") as shard:
            state.update({key: shard.get_tensor(key) for key in keys if index[key] == filename})
    return state


def _rms_zero_centered(x, weight, eps):
    return x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps) * (1 + weight.float())


def _mlp(x, state, prefix):
    gate = F.linear(x, state[prefix + "gate_proj.weight"].float())
    up = F.linear(x, state[prefix + "up_proj.weight"].float())
    return F.linear(F.silu(gate) * up, state[prefix + "down_proj.weight"].float())


def _full_attention_reference(x, normalized, layer_idx, config, cos, sin, *, causal=True):
    prefix = f"layers.{layer_idx}."
    residual = x.float()
    normed = _rms_zero_centered(x, normalized[prefix + "input_layernorm.weight"], config.rms_norm_eps)
    seq_len = x.shape[1]
    mask = None
    if causal:
        mask = torch.full((1, 1, seq_len, seq_len), torch.finfo(torch.float32).min)
        mask = torch.triu(mask, diagonal=1)
    mixed, _, _ = gated_attention_forward(
        hidden_states=normed,
        q_proj_weight=normalized[prefix + "self_attn.q_proj.weight"].float(),
        k_proj_weight=normalized[prefix + "self_attn.k_proj.weight"].float(),
        v_proj_weight=normalized[prefix + "self_attn.v_proj.weight"].float(),
        o_proj_weight=normalized[prefix + "self_attn.o_proj.weight"].float(),
        q_norm_weight=normalized[prefix + "self_attn.q_norm.weight"].float(),
        k_norm_weight=normalized[prefix + "self_attn.k_norm.weight"].float(),
        cos=cos.float(),
        sin=sin.float(),
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        attention_mask=mask,
        norm_eps=config.rms_norm_eps,
    )
    hidden = residual + mixed.float()
    normed_ff = _rms_zero_centered(hidden, normalized[prefix + "post_attention_layernorm.weight"], config.rms_norm_eps)
    return hidden + _mlp(normed_ff, normalized, prefix + "mlp.")


def _deltanet_reference(x, normalized, layer_idx, config, *, cache=None):
    prefix = f"layers.{layer_idx}."
    residual = x.float()
    normed = _rms_zero_centered(x, normalized[prefix + "input_layernorm.weight"], config.rms_norm_eps)
    qkv = normalized[prefix + "linear_attn.qkv_proj.weight"].float()
    q_dim = config.linear_num_key_heads * config.linear_key_head_dim
    if cache is None:
        batch = x.shape[0]
        q_dim = config.linear_num_key_heads * config.linear_key_head_dim
        v_dim = config.linear_num_value_heads * config.linear_value_head_dim
        cache = {
            "conv_state_q": torch.zeros(batch, q_dim, config.linear_conv_kernel_dim - 1),
            "conv_state_k": torch.zeros(batch, q_dim, config.linear_conv_kernel_dim - 1),
            "conv_state_v": torch.zeros(batch, v_dim, config.linear_conv_kernel_dim - 1),
            "recurrent_state": None,
        }
    mixed, new_cache = gated_deltanet_forward(
        hidden_states=normed,
        q_proj_weight=qkv[:q_dim],
        k_proj_weight=qkv[q_dim : 2 * q_dim],
        v_proj_weight=qkv[2 * q_dim :],
        a_proj_weight=normalized[prefix + "linear_attn.in_proj_a.weight"].float(),
        b_proj_weight=normalized[prefix + "linear_attn.in_proj_b.weight"].float(),
        o_proj_weight=normalized[prefix + "linear_attn.out_proj.weight"].float(),
        q_conv_weight=normalized[prefix + "linear_attn.q_conv.weight"].float(),
        k_conv_weight=normalized[prefix + "linear_attn.k_conv.weight"].float(),
        v_conv_weight=normalized[prefix + "linear_attn.v_conv.weight"].float(),
        q_conv_bias=None,
        k_conv_bias=None,
        v_conv_bias=None,
        A_log=normalized[prefix + "linear_attn.A_log"].float(),
        dt_bias=normalized[prefix + "linear_attn.dt_bias"].float(),
        o_norm_weight=normalized[prefix + "linear_attn.norm.weight"].float(),
        g_proj_weight=normalized[prefix + "linear_attn.in_proj_z.weight"].float(),
        num_heads=config.linear_num_key_heads,
        num_v_heads=config.linear_num_value_heads,
        head_k_dim=config.linear_key_head_dim,
        head_v_dim=config.linear_value_head_dim,
        conv_kernel_size=config.linear_conv_kernel_dim,
        norm_eps=config.rms_norm_eps,
        mode="chunk" if x.shape[1] > 1 else "fused_recurrent",
        chunk_size=128,
        conv_state_q=None if cache is None else cache["conv_state_q"],
        conv_state_k=None if cache is None else cache["conv_state_k"],
        conv_state_v=None if cache is None else cache["conv_state_v"],
        recurrent_state=None if cache is None else cache["recurrent_state"],
        output_final_state=True,
    )
    hidden = residual + mixed.float()
    normed_ff = _rms_zero_centered(hidden, normalized[prefix + "post_attention_layernorm.weight"], config.rms_norm_eps)
    return hidden + _mlp(normed_ff, normalized, prefix + "mlp."), new_cache


def _tt_rope(device, seq_len, *, start=0):
    cos, sin = compute_rope_freqs(64, start + seq_len, 10_000_000.0)
    cos_h = cos[start : start + seq_len].unsqueeze(0).to(torch.bfloat16)
    sin_h = sin[start : start + seq_len].unsqueeze(0).to(torch.bfloat16)
    return (
        cos_h,
        sin_h,
        ttnn.from_torch(cos_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        ttnn.from_torch(sin_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
    )


def _page_table(device, rows):
    return ttnn.from_torch(
        torch.tensor(rows, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )


def test_target_config_contract(hf_config):
    config = hf_config.get_text_config()
    assert config.hidden_size == 5120
    assert config.intermediate_size == 17408
    assert config.max_position_embeddings == 262144
    assert config.layer_types.count("linear_attention") == 48
    assert config.layer_types.count("full_attention") == 16


def test_runtime_fallback_source_audit():
    """Measured runtime call graph contains no host conversion or torch fallback."""
    from models.demos.blackhole.qwen36.tt import layer as layer_module
    from models.demos.blackhole.qwen36.tt.attention import decode as attention_decode
    from models.demos.blackhole.qwen36.tt.attention import prefill as attention_prefill
    from models.demos.blackhole.qwen36.tt.gdn import decode as gdn_decode

    runtime_functions = (
        FunctionalDecoder.prefill_forward,
        FunctionalDecoder.decode_forward,
        FunctionalDecoder._full_attention_forward_with_batched_paged_decode,
        FunctionalDecoder._gdn_forward_with_dram_state,
        layer_module.Qwen36DecoderLayer.forward,
        attention_prefill.prefill_forward,
        attention_decode.decode_forward,
        gdn_decode.recurrent_forward,
    )
    banned = ("torch.", "ttnn.from_torch", "ttnn.to_torch", ".cpu(", ".numpy(")
    for function in runtime_functions:
        source = inspect.getsource(function)
        assert not any(token in source for token in banned), f"runtime fallback in {function.__qualname__}"


@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_real_weights_paged_prefill_and_decode_pcc(device, hf_config):
    """Real layer-3 weights; permuted page table and tensor current position."""
    layer_idx, seq_len, block = 3, 128, 64
    config = hf_config.get_text_config()
    raw = _load_real_layer(layer_idx)
    normalized = _internal_state_dict(raw, config=config, layer_idx=layer_idx)
    decoder = FunctionalDecoder.from_state_dict(
        raw, hf_config=hf_config, layer_idx=layer_idx, mesh_device=device, max_context=256, page_block_size=block
    )
    decoder.allocate_runtime_state(batch_size=1, num_physical_blocks=4)

    torch.manual_seed(19)
    x_prefill = torch.randn(1, seq_len, config.hidden_size, dtype=torch.bfloat16) * 0.1
    cos_h, sin_h, cos_tt, sin_tt = _tt_rope(device, seq_len)
    ref_prefill = _full_attention_reference(x_prefill, normalized, layer_idx, config, cos_h, sin_h)

    page_table = _page_table(device, [[2, 0, 3, 1]])
    chunk_page_table = _page_table(device, [[2, 0]])
    x_tt = ttnn.from_torch(x_prefill, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_prefill = decoder.prefill_forward(
        x_tt,
        cos=cos_tt,
        sin=sin_tt,
        page_table=page_table,
        chunk_page_table=chunk_page_table,
        logical_seq_len=seq_len,
    )
    prefill_host = ttnn.to_torch(tt_prefill).float()
    ok, pcc = comp_pcc(ref_prefill, prefill_host, PCC_BAR)
    print(f"REAL_FULL_PREFILL_PCC={pcc}")
    assert ok, f"real-weight paged prefill PCC {pcc} < {PCC_BAR}"

    x_decode = torch.randn(1, 1, config.hidden_size, dtype=torch.bfloat16) * 0.1
    all_x = torch.cat([x_prefill, x_decode], dim=1)
    all_cos, all_sin = compute_rope_freqs(64, seq_len + 1, 10_000_000.0)
    ref_decode = _full_attention_reference(
        all_x,
        normalized,
        layer_idx,
        config,
        all_cos.unsqueeze(0).to(torch.bfloat16),
        all_sin.unsqueeze(0).to(torch.bfloat16),
    )[:, -1:]
    _, _, cos_dec, sin_dec = _tt_rope(device, 1, start=seq_len)
    pos = ttnn.from_torch(torch.tensor([seq_len], dtype=torch.int32), dtype=ttnn.int32, device=device)
    x_dec_tt = ttnn.from_torch(x_decode, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    decoder.decode_forward(x_dec_tt, cos=cos_dec, sin=sin_dec, current_position=pos, page_table=page_table)
    ttnn.synchronize_device(device)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    traced_output = decoder.decode_forward(
        x_dec_tt, cos=cos_dec, sin=sin_dec, current_position=pos, page_table=page_table
    )
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    decode_host = ttnn.to_torch(traced_output).float()
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    repeated_host = ttnn.to_torch(traced_output).float()
    ttnn.release_trace(device, trace_id)
    assert torch.equal(decode_host, repeated_host), "identical full-attention trace replay was nondeterministic"
    ok, pcc = comp_pcc(ref_decode, decode_host, PCC_BAR)
    print(f"REAL_FULL_TRACED_DECODE_PCC={pcc}")
    assert ok, f"real-weight traced paged decode PCC {pcc} < {PCC_BAR}"


@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_real_weights_deltanet_prefill_and_traced_decode_pcc(device, hf_config):
    """Real layer-0 weights; recurrent state continuity and traced in-place update."""
    layer_idx, seq_len = 0, 128
    config = hf_config.get_text_config()
    raw = _load_real_layer(layer_idx)
    normalized = _internal_state_dict(raw, config=config, layer_idx=layer_idx)
    decoder = FunctionalDecoder.from_state_dict(
        raw, hf_config=hf_config, layer_idx=layer_idx, mesh_device=device, max_context=262144
    )
    decoder.allocate_runtime_state(batch_size=1)

    torch.manual_seed(23)
    x_prefill = torch.randn(1, seq_len, config.hidden_size, dtype=torch.bfloat16) * 0.1
    ref_prefill, cache = _deltanet_reference(x_prefill, normalized, layer_idx, config)
    x_tt = ttnn.from_torch(x_prefill, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_prefill = decoder.prefill_forward(x_tt, logical_seq_len=seq_len)
    prefill_host = ttnn.to_torch(tt_prefill).float()
    ok, pcc = comp_pcc(ref_prefill, prefill_host, PCC_BAR)
    print(f"REAL_GDN_PREFILL_PCC={pcc}")
    assert ok, f"real-weight DeltaNet prefill PCC {pcc} < {PCC_BAR}"

    x_decode = torch.randn(1, 1, config.hidden_size, dtype=torch.bfloat16) * 0.1
    # Trace capture records commands but does not execute them. Eager, in-place
    # warmup, and replay are therefore three real recurrent steps.
    refs = []
    for _ in range(3):
        ref_decode, cache = _deltanet_reference(x_decode, normalized, layer_idx, config, cache=cache)
        refs.append(ref_decode)
    x_dec_tt = ttnn.from_torch(x_decode, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    eager_output = decoder.decode_forward(x_dec_tt)  # compile and materialize split convolution-state buffers
    ttnn.synchronize_device(device)
    eager_host = ttnn.to_torch(eager_output).float()
    eager_ok, eager_pcc = comp_pcc(refs[0], eager_host, PCC_BAR)
    print(f"REAL_GDN_EAGER_DECODE_PCC={eager_pcc}")
    assert eager_ok, f"real-weight eager DeltaNet decode PCC {eager_pcc} < {PCC_BAR}"
    decoder.enable_trace_safe_state_updates()
    attention = decoder.layer.attention
    before_warmup = (attention.recurrent_state.buffer_address(), attention.fused_conv_state.buffer_address())
    warmup_output = decoder.decode_forward(x_dec_tt)  # compile the in-place state-update programs used by capture
    ttnn.synchronize_device(device)
    warmup_host = ttnn.to_torch(warmup_output).float()
    after_warmup = (attention.recurrent_state.buffer_address(), attention.fused_conv_state.buffer_address())
    warmup_ok, warmup_pcc = comp_pcc(refs[1], warmup_host, PCC_BAR)
    print(f"REAL_GDN_INPLACE_DECODE_PCC={warmup_pcc}")
    assert warmup_ok, f"real-weight in-place DeltaNet decode PCC {warmup_pcc} < {PCC_BAR}"
    assert before_warmup == after_warmup
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    traced_output = decoder.decode_forward(x_dec_tt)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)
    after_capture = (attention.recurrent_state.buffer_address(), attention.fused_conv_state.buffer_address())
    assert after_warmup == after_capture
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    decode_host = ttnn.to_torch(traced_output).float()
    after_replay = (attention.recurrent_state.buffer_address(), attention.fused_conv_state.buffer_address())
    assert after_capture == after_replay
    ttnn.release_trace(device, trace_id)
    ok, pcc = comp_pcc(refs[2], decode_host, PCC_BAR)
    print(f"REAL_GDN_TRACED_DECODE_PCC={pcc}")
    assert ok, f"real-weight traced DeltaNet decode PCC {pcc} < {PCC_BAR}"


@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_full_attention_batched_paged_prefill_decode_pcc(device, hf_config):
    """Real B=2 prefill/decode with distinct lengths and independent HF row oracles."""
    config, block, padded_len, batch = hf_config.get_text_config(), 64, 96, 2
    user_lengths = [64, 96]
    layer_idx = 3
    raw = _load_real_layer(3)
    normalized = _internal_state_dict(raw, config=config, layer_idx=layer_idx)
    decoder = FunctionalDecoder.from_state_dict(
        raw, hf_config=hf_config, layer_idx=layer_idx, mesh_device=device, max_context=256, page_block_size=block
    )
    decoder.allocate_runtime_state(batch_size=batch, num_physical_blocks=8)
    rows = [[6, 2, 4, 0], [7, 3, 5, 1]]
    page_table = _page_table(device, rows)
    torch.manual_seed(37)
    x_prefill = torch.randn(batch, padded_len, config.hidden_size, dtype=torch.bfloat16) * 0.1
    cos_h, sin_h, cos_pf, sin_pf = _tt_rope(device, padded_len)
    x_tt = ttnn.from_torch(x_prefill, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    prefill = decoder.prefill_forward(
        x_tt,
        cos=cos_pf,
        sin=sin_pf,
        page_table=page_table,
        chunk_page_table=_page_table(device, [row[:2] for row in rows]),
        logical_seq_len=padded_len,
    )
    prefill_host = ttnn.to_torch(prefill).float()
    prefill_refs = []
    for user, length in enumerate(user_lengths):
        ref = _full_attention_reference(
            x_prefill[user : user + 1, :length],
            normalized,
            layer_idx,
            config,
            cos_h[:, :length],
            sin_h[:, :length],
        )
        prefill_refs.append(ref)
        ok, pcc = comp_pcc(ref, prefill_host[user : user + 1, :length], PCC_BAR)
        print(f"REAL_FULL_BATCH2_PREFILL_USER{user}_LEN{length}_PCC={pcc}")
        assert ok, f"batched prefill user {user} PCC {pcc} < {PCC_BAR}"

    x_decode = torch.randn(batch, 1, config.hidden_size, dtype=torch.bfloat16) * 0.1
    x_dec_tt = ttnn.from_torch(x_decode, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    rope_cos, rope_sin = compute_rope_freqs(64, max(user_lengths) + 1, 10_000_000.0)
    cos_dec_h = torch.stack([rope_cos[length] for length in user_lengths]).to(torch.bfloat16)
    sin_dec_h = torch.stack([rope_sin[length] for length in user_lengths]).to(torch.bfloat16)
    cos_dec = ttnn.from_torch(cos_dec_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sin_dec = ttnn.from_torch(sin_dec_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    positions = ttnn.from_torch(torch.tensor(user_lengths, dtype=torch.int32), dtype=ttnn.int32, device=device)
    output = decoder.decode_forward(
        x_dec_tt, cos=cos_dec, sin=sin_dec, current_position=positions, page_table=page_table
    )
    host = ttnn.to_torch(output).float()
    assert list(host.shape) == [batch, 1, config.hidden_size]
    assert torch.isfinite(host).all()
    for user, length in enumerate(user_lengths):
        oracle_input = torch.cat([x_prefill[user : user + 1, :length], x_decode[user : user + 1]], dim=1)
        ref = _full_attention_reference(
            oracle_input,
            normalized,
            layer_idx,
            config,
            rope_cos[: length + 1].unsqueeze(0).to(torch.bfloat16),
            rope_sin[: length + 1].unsqueeze(0).to(torch.bfloat16),
        )[:, -1:]
        ok, pcc = comp_pcc(ref, host[user : user + 1], PCC_BAR)
        print(f"REAL_FULL_BATCH2_DECODE_USER{user}_POS{length}_PCC={pcc}")
        assert ok, f"batched decode user {user} PCC {pcc} < {PCC_BAR}"


@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_deltanet_batch_two_state(device, hf_config):
    """DeltaNet state and residual shapes are not hard-coded to batch one."""
    config = hf_config.get_text_config()
    decoder = FunctionalDecoder.from_state_dict(
        _load_real_layer(0), hf_config=hf_config, layer_idx=0, mesh_device=device
    )
    decoder.allocate_runtime_state(batch_size=2)
    x = torch.randn(2, 128, config.hidden_size, dtype=torch.bfloat16) * 0.1
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    prefill = decoder.prefill_forward(x_tt, logical_seq_len=128)
    assert list(prefill.shape) == [2, 128, config.hidden_size]
    decode_x = ttnn.from_torch(
        torch.randn(2, 1, config.hidden_size, dtype=torch.bfloat16) * 0.1,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    decode = decoder.decode_forward(decode_x)
    assert list(decode.shape) == [2, 1, config.hidden_size]
    assert torch.isfinite(ttnn.to_torch(decode)).all()


@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_deltanet_repeated_input_reset_determinism(device, hf_config):
    """Resetting recurrent state reproduces the same prefill and decode bit-for-bit."""
    config = hf_config.get_text_config()
    decoder = FunctionalDecoder.from_state_dict(
        _load_real_layer(0), hf_config=hf_config, layer_idx=0, mesh_device=device
    )
    torch.manual_seed(43)
    x_prefill = torch.randn(1, 128, config.hidden_size, dtype=torch.bfloat16) * 0.1
    x_decode = torch.randn(1, 1, config.hidden_size, dtype=torch.bfloat16) * 0.1
    x_prefill_tt = ttnn.from_torch(x_prefill, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    x_decode_tt = ttnn.from_torch(x_decode, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    runs = []
    for _ in range(2):
        decoder.allocate_runtime_state(batch_size=1)
        prefill = decoder.prefill_forward(x_prefill_tt, logical_seq_len=128)
        decode = decoder.decode_forward(x_decode_tt)
        ttnn.synchronize_device(device)
        runs.append((ttnn.to_torch(prefill), ttnn.to_torch(decode)))
    assert torch.equal(runs[0][0], runs[1][0]), "DeltaNet reset prefill was nondeterministic"
    assert torch.equal(runs[0][1], runs[1][1]), "DeltaNet reset decode was nondeterministic"


@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("seq_len", [1, 31, 32, 33, 127, 128, 129])
def test_deltanet_non_aligned_logical_lengths(device, hf_config, seq_len):
    """Public prefill owns padding around tile and 128-token GDN boundaries."""
    config = hf_config.get_text_config()
    raw = _load_real_layer(0)
    decoder = FunctionalDecoder.from_state_dict(raw, hf_config=hf_config, layer_idx=0, mesh_device=device)
    decoder.allocate_runtime_state(batch_size=1)
    x = torch.randn(1, seq_len, config.hidden_size, dtype=torch.bfloat16) * 0.1
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = decoder.prefill_forward(x_tt, logical_seq_len=seq_len)
    assert list(out.shape) == [1, seq_len, config.hidden_size]
    assert torch.isfinite(ttnn.to_torch(out)).all()


@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("seq_len", [31, 32, 33, 63, 64, 65, 127, 128, 129])
def test_full_attention_non_aligned_paged_prefill(device, hf_config, seq_len):
    """Paged prefill accepts logical lengths around tile/page/chunk boundaries."""
    config, block = hf_config.get_text_config(), 64
    raw = _load_real_layer(3)
    decoder = FunctionalDecoder.from_state_dict(
        raw, hf_config=hf_config, layer_idx=3, mesh_device=device, max_context=256, page_block_size=block
    )
    blocks = (seq_len + block - 1) // block
    decoder.allocate_runtime_state(batch_size=1, num_physical_blocks=max(4, blocks))
    x = torch.randn(1, seq_len, config.hidden_size, dtype=torch.bfloat16) * 0.1
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    _, _, cos_tt, sin_tt = _tt_rope(device, seq_len)
    physical = list(reversed(range(max(4, blocks))))
    page_table = _page_table(device, [physical])
    chunk_page_table = _page_table(device, [physical[:blocks]])
    out = decoder.prefill_forward(
        x_tt,
        cos=cos_tt,
        sin=sin_tt,
        page_table=page_table,
        chunk_page_table=chunk_page_table,
        logical_seq_len=seq_len,
    )
    assert list(out.shape) == [1, seq_len, config.hidden_size]
    assert torch.isfinite(ttnn.to_torch(out)).all()


@run_for_blackhole()
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_advertised_context_layer_harness(device, hf_config):
    """Reach decode position 262,143 through a non-aligned final prefill.

    This is deliberately a reduced layer harness: outputs are consumed and
    deallocated chunk-by-chunk, as a stacked model would do, rather than keeping
    every layer output resident simultaneously. The 262,143-token prefill
    forces both a partial 127-token DeltaNet chunk and a partial final KV page.
    """
    config = hf_config.get_text_config()
    context, block = config.max_position_embeddings, 64
    prefill_tokens = context - 1

    # DeltaNet has a fixed-size recurrent state; its kernel contract is 128 rows.
    gdn = FunctionalDecoder.from_state_dict(
        _load_real_layer(0), hf_config=hf_config, layer_idx=0, mesh_device=device, max_context=context
    )
    gdn.allocate_runtime_state(batch_size=1)
    gdn_chunk = torch.zeros(1, 128, config.hidden_size, dtype=torch.bfloat16)
    gdn_chunk_tt = ttnn.from_torch(gdn_chunk, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    for _ in range(prefill_tokens // 128):
        output = gdn.prefill_forward(gdn_chunk_tt, logical_seq_len=128)
        ttnn.deallocate(output)
    gdn_tail_len = prefill_tokens % 128
    gdn_tail = ttnn.from_torch(
        torch.zeros(1, gdn_tail_len, config.hidden_size, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    output = gdn.prefill_forward(gdn_tail, logical_seq_len=gdn_tail_len)
    assert list(output.shape) == [1, gdn_tail_len, config.hidden_size]
    ttnn.deallocate(output)
    gdn_decode = ttnn.from_torch(
        torch.zeros(1, 1, config.hidden_size, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    gdn_out = gdn.decode_forward(gdn_decode)
    assert torch.isfinite(ttnn.to_torch(gdn_out)).all()

    # Full attention uses paged KV and 2K chunks; the final decode reads the
    # full 4096-entry logical page table at the last advertised position.
    full = FunctionalDecoder.from_state_dict(
        _load_real_layer(3),
        hf_config=hf_config,
        layer_idx=3,
        mesh_device=device,
        max_context=context,
        page_block_size=block,
    )
    num_blocks = context // block
    full.allocate_runtime_state(batch_size=1, num_physical_blocks=num_blocks)
    physical = torch.arange(num_blocks - 1, -1, -1, dtype=torch.int32).unsqueeze(0)
    full_page_table = ttnn.from_torch(physical, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    prefill_chunk = 2048
    x_chunk = ttnn.from_torch(
        torch.zeros(1, prefill_chunk, config.hidden_size, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    cos_all, sin_all = compute_rope_freqs(64, context, 10_000_000.0)
    for start in range(0, prefill_tokens, prefill_chunk):
        logical_len = min(prefill_chunk, prefill_tokens - start)
        blocks_this_chunk = (logical_len + block - 1) // block
        block_start = start // block
        chunk_pt = ttnn.from_torch(
            physical[:, block_start : block_start + blocks_this_chunk].contiguous(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        cos = ttnn.from_torch(
            cos_all[start : start + logical_len].unsqueeze(0).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        sin = ttnn.from_torch(
            sin_all[start : start + logical_len].unsqueeze(0).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        start_tt = ttnn.from_torch(torch.tensor([start], dtype=torch.int32), dtype=ttnn.int32, device=device)
        chunk_input = x_chunk
        if logical_len != prefill_chunk:
            chunk_input = ttnn.from_torch(
                torch.zeros(1, logical_len, config.hidden_size, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
        output = full.prefill_forward(
            chunk_input,
            cos=cos,
            sin=sin,
            page_table=full_page_table,
            chunk_page_table=chunk_pt,
            chunk_start_idx_tensor=start_tt,
            logical_seq_len=logical_len,
        )
        ttnn.deallocate(output)
        ttnn.deallocate(chunk_pt)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        ttnn.deallocate(start_tt)
    pos = prefill_tokens
    cos = ttnn.from_torch(
        cos_all[pos : pos + 1].unsqueeze(0).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    sin = ttnn.from_torch(
        sin_all[pos : pos + 1].unsqueeze(0).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    position = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32, device=device)
    decode_x = ttnn.from_torch(
        torch.zeros(1, 1, config.hidden_size, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    full_out = full.decode_forward(decode_x, cos=cos, sin=sin, current_position=position, page_table=full_page_table)
    assert torch.isfinite(ttnn.to_torch(full_out)).all()
    print(
        f"ADVERTISED_CONTEXT_NONALIGNED_PREFILL={prefill_tokens} "
        f"ADVERTISED_CONTEXT_DECODE_POSITION={pos} "
        f"GDN_FINAL_CHUNK={gdn_tail_len} FULL_FINAL_CHUNK={prefill_tokens % prefill_chunk}"
    )


@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("profile_case", ["full_prefill", "full_decode", "gdn_prefill", "gdn_decode"])
def test_profile_warmed_prefill_and_traced_decode(device, hf_config, profile_case):
    """Profiler entry point; select exactly one bounded measured window per process."""
    from tracy import signpost

    config, seq_len, block = hf_config.get_text_config(), 128, 64
    x_pf_h = torch.zeros(1, seq_len, config.hidden_size, dtype=torch.bfloat16)
    x_dec_h = torch.zeros(1, 1, config.hidden_size, dtype=torch.bfloat16)

    if profile_case.startswith("full_"):
        full = FunctionalDecoder.from_state_dict(
            _load_real_layer(3), hf_config=hf_config, layer_idx=3, mesh_device=device, max_context=256
        )
        full.allocate_runtime_state(batch_size=1, num_physical_blocks=4)
        pt = _page_table(device, [[2, 0, 3, 1]])
        cpt = _page_table(device, [[2, 0]])
        _, _, cos_pf, sin_pf = _tt_rope(device, seq_len)
        x_pf = ttnn.from_torch(x_pf_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        full.prefill_forward(x_pf, cos=cos_pf, sin=sin_pf, page_table=pt, chunk_page_table=cpt)
        ttnn.synchronize_device(device)
        if profile_case == "full_prefill":
            signpost("PERF_FULL_PREFILL")
            full.prefill_forward(x_pf, cos=cos_pf, sin=sin_pf, page_table=pt, chunk_page_table=cpt)
            ttnn.synchronize_device(device)
            signpost("PERF_FULL_PREFILL_END")
            return
        _, _, cos_d, sin_d = _tt_rope(device, 1, start=seq_len)
        pos = ttnn.from_torch(torch.tensor([seq_len], dtype=torch.int32), dtype=ttnn.int32, device=device)
        x_dec = ttnn.from_torch(x_dec_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        full.decode_forward(x_dec, cos=cos_d, sin=sin_d, current_position=pos, page_table=pt)
        ttnn.synchronize_device(device)
        full_tid = ttnn.begin_trace_capture(device, cq_id=0)
        full.decode_forward(x_dec, cos=cos_d, sin=sin_d, current_position=pos, page_table=pt)
        ttnn.end_trace_capture(device, full_tid, cq_id=0)
        ttnn.synchronize_device(device)
        ttnn.execute_trace(device, full_tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        signpost("PERF_FULL_DECODE")
        ttnn.execute_trace(device, full_tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        signpost("PERF_FULL_DECODE_END")
        ttnn.release_trace(device, full_tid)
        return

    # DeltaNet: chunk prefill and traced recurrent decode.
    gdn = FunctionalDecoder.from_state_dict(_load_real_layer(0), hf_config=hf_config, layer_idx=0, mesh_device=device)
    gdn.allocate_runtime_state(batch_size=1)
    gdn_pf = ttnn.from_torch(x_pf_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    gdn.prefill_forward(gdn_pf, logical_seq_len=seq_len)
    ttnn.synchronize_device(device)
    if profile_case == "gdn_prefill":
        signpost("PERF_GDN_PREFILL")
        gdn.prefill_forward(gdn_pf, logical_seq_len=seq_len)
        ttnn.synchronize_device(device)
        signpost("PERF_GDN_PREFILL_END")
        return
    gdn_dec = ttnn.from_torch(x_dec_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    gdn.decode_forward(gdn_dec)
    ttnn.synchronize_device(device)
    gdn.enable_trace_safe_state_updates()
    gdn.decode_forward(gdn_dec)
    ttnn.synchronize_device(device)
    gdn_tid = ttnn.begin_trace_capture(device, cq_id=0)
    gdn.decode_forward(gdn_dec)
    ttnn.end_trace_capture(device, gdn_tid, cq_id=0)
    ttnn.synchronize_device(device)
    ttnn.execute_trace(device, gdn_tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    signpost("PERF_GDN_DECODE")
    ttnn.execute_trace(device, gdn_tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    signpost("PERF_GDN_DECODE_END")
    ttnn.release_trace(device, gdn_tid)
