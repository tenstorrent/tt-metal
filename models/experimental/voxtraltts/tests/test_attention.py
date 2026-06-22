# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.functional import (
    VoxtralTextConfig,
    compute_rope_frequencies,
    text_attention as reference_text_attention,
)
from models.experimental.voxtraltts.reference.voxtral_config import load_voxtral_config
from models.experimental.voxtraltts.utils.test_common import (
    load_acoustic_fm_layer_weights_or_skip,
    load_audio_tokenizer_state_dict_or_skip,
    resolve_voxtral_model_name_or_skip,
)
from models.experimental.voxtraltts.tt.attention import VoxtralTTAttention


@torch.no_grad()
def test_voxtral_text_attention_pcc(device, reset_seeds):
    """GQA attention PCC at production FM dims (3072 hidden, 32/8 heads) with checkpoint weights."""
    model_name = resolve_voxtral_model_name_or_skip()
    layer_weights = load_acoustic_fm_layer_weights_or_skip(0)
    ac_cfg = load_voxtral_config(model_name).audio_model_args.acoustic_transformer_args

    hidden = ac_cfg.dim
    n_heads = ac_cfg.n_heads
    n_kv_heads = ac_cfg.n_kv_heads
    head_dim = ac_cfg.head_dim
    seq_len = 64

    config = VoxtralTextConfig(
        hidden_size=hidden,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        head_dim=head_dim,
        rope_theta=ac_cfg.rope_theta,
    )

    torch.manual_seed(0)
    torch_input = torch.randn(1, 1, seq_len, hidden, dtype=torch.bfloat16)

    cos, sin = compute_rope_frequencies(
        head_dim=head_dim,
        max_seq_len=seq_len,
        theta=config.rope_theta,
        device=torch.device("cpu"),
    )
    reference_output = reference_text_attention(
        hidden_states=torch_input.squeeze(1),
        layer_weights=layer_weights,
        cos=cos,
        sin=sin,
        config=config,
        attention_mask=None,
    )

    tt_model = VoxtralTTAttention(
        device=device,
        hidden_size=hidden,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        head_dim=head_dim,
        state_dict=layer_weights,
        weight_prefix="attention",
        activation_memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    tt_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_output = tt_model(tt_input, cos=cos, sin=sin)
    tt_output_torch = ttnn.to_torch(tt_output).squeeze(1)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc=0.985)
    assert passing, f"Voxtral FM attention PCC failed: {pcc_message}"


@torch.no_grad()
def test_voxtral_attention_causal_identity_rope_runs(device, reset_seeds):
    """Causal SDPA path (audio tokenizer): checkpoint weights, identity cos/sin — shape + no throw."""
    model_name = resolve_voxtral_model_name_or_skip()
    cfg = load_voxtral_config(model_name).audio_tokenizer_args
    sd = load_audio_tokenizer_state_dict_or_skip()

    block_index, layer_index = 1, 0
    prefix = f"decoder_blocks.{block_index}.layers.{layer_index}.attention"
    if f"{prefix}.wq.weight" not in sd:
        pytest.skip(f"Missing {prefix}.wq.weight in checkpoint")

    batch = 1
    seq_len = 32
    hidden = cfg.dim
    n_heads = cfg.n_heads
    n_kv_heads = cfg.n_kv_heads
    head_dim = cfg.head_dim

    torch.manual_seed(0)
    torch_input = torch.randn(batch, 1, seq_len, hidden, dtype=torch.bfloat16)
    cos = torch.ones(batch, seq_len, head_dim, dtype=torch.bfloat16)
    sin = torch.zeros(batch, seq_len, head_dim, dtype=torch.bfloat16)

    tt_model = VoxtralTTAttention(
        device=device,
        hidden_size=hidden,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        head_dim=head_dim,
        state_dict=sd,
        weight_prefix=prefix,
        is_causal=True,
        use_qk_norm=cfg.qk_norm,
        qk_norm_eps=cfg.qk_norm_eps,
        activation_memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_output = tt_model(tt_input, cos=cos, sin=sin)
    out_t = ttnn.to_torch(tt_output)
    assert out_t.shape == torch_input.shape
