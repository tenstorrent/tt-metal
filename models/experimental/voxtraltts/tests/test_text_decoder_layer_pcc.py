# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device PCC for text decoder layers (local ``text_attention`` / ``text_mlp``).

Uses ``inner.layers[i]`` from ``VoxtralTTTextModel`` — the same production blocks
with local attention/MLP swapped in at construction.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.functional import (
    VoxtralTextConfig,
    compute_rope_frequencies as reference_compute_rope_frequencies,
    extract_layer_weights,
    text_decoder_layer as reference_text_decoder_layer,
)
from models.experimental.voxtraltts.tests.common import create_real_voxtral_text_model_or_skip
from models.experimental.voxtraltts.tt.voxtral_tt_args import voxtral_text_logits_pcc_optimizations
from models.tt_transformers.tt.common import Mode

PCC_TARGET = 0.99


def _ref_config(args) -> VoxtralTextConfig:
    return VoxtralTextConfig(
        hidden_size=args.dim,
        num_hidden_layers=args.n_layers,
        num_attention_heads=args.n_heads,
        num_key_value_heads=args.n_kv_heads,
        head_dim=args.head_dim,
        intermediate_size=args.hidden_dim,
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_seq_len,
        rope_theta=args.rope_theta,
        rms_norm_eps=args.norm_eps,
    )


def _reference_layer_output(state_dict, args, hidden_in: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """Run reference layer ``layer_idx`` over a full causal sequence -> [1, S, dim]."""
    seq_len = hidden_in.shape[1]
    cfg = _ref_config(args)
    cos, sin = reference_compute_rope_frequencies(
        head_dim=cfg.head_dim,
        max_seq_len=seq_len,
        theta=cfg.rope_theta,
        device=hidden_in.device,
    )
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), dtype=torch.float32)
    mask = torch.triu(mask, diagonal=1)
    weights = extract_layer_weights(state_dict, layer_idx, prefix="layers.")
    return reference_text_decoder_layer(
        hidden_states=hidden_in,
        layer_weights=weights,
        cos=cos,
        sin=sin,
        config=cfg,
        attention_mask=mask,
    )


def _hidden_4d_to_torch(inner, tt_x: ttnn.Tensor, last_token_idx: int | None = None) -> torch.Tensor:
    host = inner.concat_host_output(tt_x)
    dim = inner.args.dim
    idx = last_token_idx if last_token_idx is not None else host.shape[2] - 1
    return host[0, 0, idx, :dim].to(dtype=torch.float32)


def _run_layer_prefill_pcc(model, layer_idx: int, *, seq_len: int = 128) -> float:
    inner = model.inner
    args = inner.args
    state_dict = args.load_state_dict()
    layer = inner.layers[layer_idx]
    dim = args.dim

    hidden_in = (torch.randn(1, seq_len, dim, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    ref_out = _reference_layer_output(state_dict, args, hidden_in, layer_idx)

    dummy_tokens = torch.zeros(1, seq_len, dtype=torch.int64)
    _, rot_global, rot_local, _, _, _ = model.prepare_inputs_prefill(dummy_tokens, start_pos=0)
    prefill_cfg = args.get_residual_mem_config(Mode.PREFILL, inner.prefetcher)
    x_tt = ttnn.from_torch(
        hidden_in.reshape(1, 1, seq_len, dim).to(torch.bfloat16),
        device=inner.mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=prefill_cfg,
        mesh_mapper=ttnn.ReplicateTensorToMesh(inner.mesh_device),
    )
    tt_out = layer(
        x_tt,
        None,
        rot_mats_global=rot_global,
        rot_mats_local=rot_local,
        user_id=0,
        mode=Mode.PREFILL,
        page_table=None,
        kv_cache=None,
    )
    tt_last = _hidden_4d_to_torch(inner, tt_out, last_token_idx=seq_len - 1)
    ref_last = ref_out[0, -1, :].to(torch.float32)
    passing, pcc = comp_pcc(ref_last, tt_last, pcc=PCC_TARGET)
    assert passing, f"Prefill layer-{layer_idx} hidden mismatch vs reference: {pcc}"
    return float(pcc)


def _run_layer_decode_pcc(model, layer_idx: int, *, prompt_len: int = 128) -> float:
    inner = model.inner
    args = inner.args
    state_dict = args.load_state_dict()
    layer = inner.layers[layer_idx]
    dim = args.dim

    hidden_full = (torch.randn(1, prompt_len + 1, dim, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    ref_out = _reference_layer_output(state_dict, args, hidden_full, layer_idx)
    ref_last = ref_out[0, -1, :].to(torch.float32)

    prompt_hidden = hidden_full[:, :prompt_len, :]
    dummy_tokens = torch.zeros(1, prompt_len, dtype=torch.int64)
    _, rot_global_p, rot_local_p, _, _, _ = model.prepare_inputs_prefill(dummy_tokens, start_pos=0)
    prefill_cfg = args.get_residual_mem_config(Mode.PREFILL, inner.prefetcher)
    x_prefill = ttnn.from_torch(
        prompt_hidden.reshape(1, 1, prompt_len, dim).to(torch.bfloat16),
        device=inner.mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=prefill_cfg,
        mesh_mapper=ttnn.ReplicateTensorToMesh(inner.mesh_device),
    )
    _ = layer(
        x_prefill,
        None,
        rot_mats_global=rot_global_p,
        rot_mats_local=rot_local_p,
        user_id=0,
        mode=Mode.PREFILL,
        page_table=None,
        kv_cache=None,
    )

    pos_idx = prompt_len
    current_pos_t = torch.tensor([pos_idx], dtype=torch.int64)
    rot_global_d = inner.rope_setup.get_rot_mats(current_pos_t)
    rot_local_d = inner.rope_local_setup.get_rot_mats(current_pos_t) if hasattr(inner, "rope_local_setup") else None
    current_pos_tt = ttnn.from_torch(
        current_pos_t,
        device=inner.mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(inner.mesh_device, dims=(None, None), mesh_shape=args.cluster_shape),
    )
    decode_cfg = args.get_residual_mem_config(Mode.DECODE, inner.prefetcher)
    new_hidden = hidden_full[:, prompt_len, :]
    x_decode = ttnn.from_torch(
        new_hidden.reshape(1, 1, 1, dim).to(torch.bfloat16),
        device=inner.mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=decode_cfg,
        mesh_mapper=ttnn.ReplicateTensorToMesh(inner.mesh_device),
    )
    tt_out = layer(
        x_decode,
        current_pos_tt,
        rot_mats_global=rot_global_d,
        rot_mats_local=rot_local_d,
        user_id=0,
        mode=Mode.DECODE,
        page_table=None,
        kv_cache=None,
    )
    tt_last = _hidden_4d_to_torch(inner, tt_out, last_token_idx=0)
    passing, pcc = comp_pcc(ref_last, tt_last, pcc=PCC_TARGET)
    assert passing, f"Decode layer-{layer_idx} hidden mismatch vs reference: {pcc}"
    return float(pcc)


@torch.no_grad()
@pytest.mark.timeout(3600)
def test_text_decoder_layer_prefill_pcc(device, reset_seeds):
    """Layer 0 prefill PCC (fast smoke)."""
    model = create_real_voxtral_text_model_or_skip(
        device, max_seq_len=256, dtype=ttnn.bfloat16, optimizations=voxtral_text_logits_pcc_optimizations
    )
    pcc = _run_layer_prefill_pcc(model, 0)
    logger.info(f"test_text_decoder_layer_prefill_pcc layer=0 PCC={pcc:.6f}")


@torch.no_grad()
@pytest.mark.timeout(3600)
def test_text_decoder_layer_decode_pcc(device, reset_seeds):
    """Layer 0 decode PCC with populated KV cache (fast smoke)."""
    model = create_real_voxtral_text_model_or_skip(
        device, max_seq_len=256, dtype=ttnn.bfloat16, optimizations=voxtral_text_logits_pcc_optimizations
    )
    pcc = _run_layer_decode_pcc(model, 0)
    logger.info(f"test_text_decoder_layer_decode_pcc layer=0 PCC={pcc:.6f}")


@torch.no_grad()
@pytest.mark.timeout(7200)
def test_text_decoder_layer_prefill_pcc_all_layers(device, reset_seeds):
    """Prefill PCC for every text decoder layer (``args.n_layers``, 26 for Voxtral-4B-TTS)."""
    model = create_real_voxtral_text_model_or_skip(
        device, max_seq_len=256, dtype=ttnn.bfloat16, optimizations=voxtral_text_logits_pcc_optimizations
    )
    n_layers = int(model.inner.args.n_layers)
    for layer_idx in range(n_layers):
        pcc = _run_layer_prefill_pcc(model, layer_idx)
        logger.info(f"prefill layer={layer_idx}/{n_layers - 1} PCC={pcc:.6f}")


@torch.no_grad()
@pytest.mark.timeout(7200)
def test_text_decoder_layer_decode_pcc_all_layers(device, reset_seeds):
    """Decode PCC for every text decoder layer (``args.n_layers``, 26 for Voxtral-4B-TTS)."""
    model = create_real_voxtral_text_model_or_skip(
        device, max_seq_len=256, dtype=ttnn.bfloat16, optimizations=voxtral_text_logits_pcc_optimizations
    )
    n_layers = int(model.inner.args.n_layers)
    for layer_idx in range(n_layers):
        pcc = _run_layer_decode_pcc(model, layer_idx)
        logger.info(f"decode layer={layer_idx}/{n_layers - 1} PCC={pcc:.6f}")
