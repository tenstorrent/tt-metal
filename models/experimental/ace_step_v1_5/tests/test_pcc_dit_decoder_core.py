# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch

from models.experimental.ace_step_v1_5.tests._dit_decoder_pcc_common import assert_pcc_print
from models.experimental.ace_step_v1_5.torch_ref.dit_decoder_core import (
    TorchAceStepDiTCoreRef,
    TorchTimestepEmbeddingRef,
    make_time_embed_state_dict,
    make_tiny_state_dict,
)
from models.experimental.ace_step_v1_5.ttnn_impl.dit_decoder_core import (
    AceStepDecoderConfigTTNN,
    TtAceStepDiTCore,
    TtTimestepEmbedding,
)


def test_timestep_embedding_matches_torch(mesh_device):
    import ttnn

    hidden_size = 64
    timesteps_host = np.linspace(1.0, 0.0, num=9, dtype=np.float32)  # includes 0.0
    sd = make_time_embed_state_dict(hidden_size=hidden_size)

    cfg = AceStepDecoderConfigTTNN(
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        rms_norm_eps=1e-6,
        sliding_window=None,
    )

    tt = TtTimestepEmbedding(
        cfg=cfg,
        state_dict=sd,
        base_address="time_embed",
        mesh_device=mesh_device,
        timesteps_host=timesteps_host,
        dtype=ttnn.bfloat16,
    )

    timestep_index = 3
    ref = TorchTimestepEmbeddingRef(
        hidden_size=hidden_size, state_dict=sd, base="time_embed", timesteps_host=timesteps_host
    )
    temb_ref, tp_ref = ref(timestep_index)

    temb_tt, tp_tt = tt(timestep_index)
    temb = ttnn.to_torch(temb_tt).to(torch.bfloat16)
    tp = ttnn.to_torch(tp_tt).to(torch.bfloat16)

    assert_pcc_print("timestep_embedding_temb", temb_ref, temb)
    assert_pcc_print("timestep_embedding_proj", tp_ref, tp)


def test_dit_decoder_core_matches_torch(mesh_device):
    import ttnn

    B = 1
    S = 32
    S_enc = 16
    # SDPA on device currently does not support padding the head_dim dimension in TILE layout.
    # Use a head_dim that is tile-aligned to avoid implicit padding.
    head_dim = 32
    n_heads = 4
    D = n_heads * head_dim
    cond_dim = 32
    intermediate = 256
    num_layers = 1

    cfg = AceStepDecoderConfigTTNN(
        hidden_size=D,
        num_hidden_layers=num_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,
        head_dim=head_dim,
        rms_norm_eps=1e-6,
        sliding_window=None,
    )

    sd = make_tiny_state_dict(
        d_model=D,
        n_heads=n_heads,
        head_dim=head_dim,
        cond_dim=cond_dim,
        intermediate=intermediate,
        num_layers=num_layers,
    )

    torch.manual_seed(1)
    x_patches = torch.randn(B, S, D, dtype=torch.bfloat16)
    timestep_proj = torch.randn(B, 6, D, dtype=torch.bfloat16)
    enc = torch.randn(B, S_enc, cond_dim, dtype=torch.bfloat16)

    y_ref = TorchAceStepDiTCoreRef(cfg=cfg, state_dict=sd)(x_patches, timestep_proj, enc)

    tt_core = TtAceStepDiTCore(cfg=cfg, state_dict=sd, mesh_device=mesh_device, dtype=ttnn.bfloat16)
    x_tt = ttnn.from_torch(x_patches, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    tp_tt = ttnn.from_torch(timestep_proj, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    enc_tt = ttnn.from_torch(enc, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    y_tt = tt_core(x_tt, tp_tt, enc_tt)
    y = ttnn.to_torch(y_tt).to(torch.bfloat16)

    assert_pcc_print("dit_decoder_core_mha", y_ref, y)


def test_dit_decoder_core_gqa_matches_torch(mesh_device):
    """GQA (n_kv < n_heads): KV expansion must use repeat_interleave, not tile-style repeat."""
    import ttnn

    B = 1
    S = 32
    S_enc = 16
    head_dim = 32
    n_heads = 4
    n_kv = 2
    D = n_heads * head_dim
    cond_dim = 32
    intermediate = 256
    num_layers = 1

    cfg = AceStepDecoderConfigTTNN(
        hidden_size=D,
        num_hidden_layers=num_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        head_dim=head_dim,
        rms_norm_eps=1e-6,
        sliding_window=None,
    )

    sd = make_tiny_state_dict(
        d_model=D,
        n_heads=n_heads,
        head_dim=head_dim,
        cond_dim=cond_dim,
        intermediate=intermediate,
        num_layers=num_layers,
        n_kv_heads=n_kv,
    )

    torch.manual_seed(2)
    x_patches = torch.randn(B, S, D, dtype=torch.bfloat16)
    timestep_proj = torch.randn(B, 6, D, dtype=torch.bfloat16)
    enc = torch.randn(B, S_enc, cond_dim, dtype=torch.bfloat16)

    y_ref = TorchAceStepDiTCoreRef(cfg=cfg, state_dict=sd)(x_patches, timestep_proj, enc)

    tt_core = TtAceStepDiTCore(cfg=cfg, state_dict=sd, mesh_device=mesh_device, dtype=ttnn.bfloat16)
    x_tt = ttnn.from_torch(x_patches, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    tp_tt = ttnn.from_torch(timestep_proj, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    enc_tt = ttnn.from_torch(enc, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    y_tt = tt_core(x_tt, tp_tt, enc_tt)
    y = ttnn.to_torch(y_tt).to(torch.bfloat16)

    assert_pcc_print("dit_decoder_core_gqa", y_ref, y)
