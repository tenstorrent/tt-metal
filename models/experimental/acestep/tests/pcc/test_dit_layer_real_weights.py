# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: ACE-Step v1.5 DiT layer with REAL checkpoint weights.

All other tests use random weights, which can hide dtype/scale issues that only surface with
the trained bf16 distribution (per-channel outliers, weight magnitude, q/k-norm scale). This
test loads the genuine `decoder.layers.1` tensors from model.safetensors and validates the TT
AceStepDiTLayer against the real reference module — the strongest correctness signal in the suite.

Requires model.safetensors (4.79 GB) in the HF cache; skips cleanly if absent.
"""

import pytest
import torch

import ttnn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from models.experimental.acestep.reference.hf_reference import load_config, load_modeling_module
from models.experimental.acestep.reference.weight_utils import checkpoint_path, load_module_weights
from models.experimental.acestep.tt.dit_layer import AceStepDiTLayer, AceStepDiTLayerConfig
from models.experimental.acestep.tests.test_utils import (
    HEAD_DIM,
    HIDDEN_SIZE,
    NUM_ATTENTION_HEADS,
    NUM_KEY_VALUE_HEADS,
    RMS_NORM_EPS,
    assert_pcc,
    make_lazy_weight,
    require_single_device,
    to_torch,
    to_ttnn_tensor,
)

REAL_LAYER_IDX = 1  # full_attention
SHAPES = [(256, 128), (512, 256)]


def _have_checkpoint():
    try:
        checkpoint_path()
        return True
    except AssertionError:
        return False


def _wT(w, d):
    return make_lazy_weight(
        w.detach().clone().transpose(-1, -2).contiguous(), d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )


def _norm(w, d):
    return make_lazy_weight(w.detach().clone(), d, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)


@pytest.mark.skipif(not _have_checkpoint(), reason="model.safetensors not downloaded")
@pytest.mark.parametrize("seq_len,enc_len", SHAPES, ids=[f"S{s}_E{e}" for s, e in SHAPES])
def test_dit_layer_real_weights_vs_hf(device, seq_len, enc_len):
    require_single_device(device)
    torch.manual_seed(0)

    m = load_modeling_module()
    cfg = load_config()
    cfg._attn_implementation = "eager"
    ref = m.AceStepDiTLayer(cfg, layer_idx=REAL_LAYER_IDX, use_cross_attention=True).eval()
    # Load genuine trained weights for this layer.
    load_module_weights(ref, f"decoder.layers.{REAL_LAYER_IDX}.")

    # Realistic-scale inputs (unit normal; the layer's own norms handle scaling).
    hidden = torch.randn(1, seq_len, HIDDEN_SIZE, dtype=torch.float32)
    encoder = torch.randn(1, enc_len, HIDDEN_SIZE, dtype=torch.float32)
    temb = torch.randn(1, 6, HIDDEN_SIZE, dtype=torch.float32)

    rope = Qwen3RotaryEmbedding(cfg)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = rope(hidden, position_ids)

    with torch.no_grad():
        (ref_out,) = ref(
            hidden_states=hidden,
            position_embeddings=(cos, sin),
            temb=temb,
            encoder_hidden_states=encoder,
            attention_mask=None,
            encoder_attention_mask=None,
        )

    a, c = ref.self_attn, ref.cross_attn
    tt = AceStepDiTLayer(
        AceStepDiTLayerConfig(
            scale_shift_table=make_lazy_weight(
                ref.scale_shift_table.detach().clone(), device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            ),
            self_attn_norm_weight=_norm(ref.self_attn_norm.weight, device),
            mlp_norm_weight=_norm(ref.mlp_norm.weight, device),
            cross_attn_norm_weight=_norm(ref.cross_attn_norm.weight, device),
            wq=_wT(a.q_proj.weight, device),
            wk=_wT(a.k_proj.weight, device),
            wv=_wT(a.v_proj.weight, device),
            wo=_wT(a.o_proj.weight, device),
            q_norm_weight=_norm(a.q_norm.weight, device),
            k_norm_weight=_norm(a.k_norm.weight, device),
            c_wq=_wT(c.q_proj.weight, device),
            c_wk=_wT(c.k_proj.weight, device),
            c_wv=_wT(c.v_proj.weight, device),
            c_wo=_wT(c.o_proj.weight, device),
            c_q_norm_weight=_norm(c.q_norm.weight, device),
            c_k_norm_weight=_norm(c.k_norm.weight, device),
            w1=_wT(ref.mlp.gate_proj.weight, device),
            w2=_wT(ref.mlp.down_proj.weight, device),
            w3=_wT(ref.mlp.up_proj.weight, device),
            n_heads=NUM_ATTENTION_HEADS,
            n_kv_heads=NUM_KEY_VALUE_HEADS,
            head_dim=HEAD_DIM,
            dim=HIDDEN_SIZE,
            eps=RMS_NORM_EPS,
            sliding_window=None,
            use_cross_attention=True,
        )
    )

    hidden_tt = to_ttnn_tensor(hidden.reshape(1, 1, seq_len, HIDDEN_SIZE), device)
    encoder_tt = to_ttnn_tensor(encoder.reshape(1, 1, enc_len, HIDDEN_SIZE), device)
    cos_tt = ttnn.from_torch(cos.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sin_tt = ttnn.from_torch(sin.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    temb_tt = ttnn.from_torch(
        temb.reshape(1, 6, 1, HIDDEN_SIZE), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )

    out_tt = tt.forward(hidden_tt, cos_tt, sin_tt, temb_tt, encoder_hidden_states=encoder_tt)
    out = to_torch(out_tt, expected_shape=(1, 1, seq_len, HIDDEN_SIZE)).reshape(1, seq_len, HIDDEN_SIZE)

    assert_pcc(ref_out, out, 0.98)
