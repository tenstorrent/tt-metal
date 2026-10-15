# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device smoke tests: SDPA recipes on the D64 joint attentions of SD3.5 (SD35JointAttention) and Motif
(blocks/attention.py with context head scaling), variants and gates from sdpa_smoke.py (gate_vs_legacy).

Random weights, small shapes (4 heads, D64), 1x1 (joint SDPA) and 1x2 (ring joint SDPA, sp=2), with the
non-tile-aligned SD3.5/Motif prompt length 333; Motif also with an unaligned (padded) spatial length on 1x2.
"""

from __future__ import annotations

import diffusers.models.attention_processor
import pytest
import torch

from ...blocks.attention import Attention
from ...models.transformers.attention_sd35 import SD35JointAttention
from ...models.transformers.transformer_motif import MotifTransformer, convert_motif_attention_state
from ...reference.motif.modeling_dit import JointAttn as MotifAttentionReference
from ...utils import tensor
from ...utils.tensor import bf16_tensor
from .sdpa_smoke import MESHES, gate_vs_legacy, parallel, run_variants

HEADS = 4
HEAD_DIM = 64
DIM = HEADS * HEAD_DIM

# variant -> (absolute L2 bound %, margin over legacy % points, legacy scale)
BOUNDS = {"default": (3.0, 1.0, 1.0), "FAST_bfp8": (3.0, 1.0, 1.0)}


# ---------------------------------------------------------------------------------------------
# SD3.5 joint attention (models/transformers/attention_sd35.py)
# ---------------------------------------------------------------------------------------------


@MESHES
@pytest.mark.parametrize(("spatial_seq_len", "prompt_seq_len"), [pytest.param(1024, 333, id="s1024_p333")])
def test_sd35_attention_sdpa_recipes(
    mesh_device, sp_axis, tp_axis, spatial_seq_len, prompt_seq_len, record_property
) -> None:
    torch.manual_seed(0)
    ccl_manager, parallel_config, _ = parallel(mesh_device, sp_axis, tp_axis)

    # As in diffusers' SD3 JointTransformerBlock (not the last block: prompt output is produced).
    torch_model = diffusers.models.attention_processor.Attention(
        query_dim=DIM,
        cross_attention_dim=None,
        added_kv_proj_dim=DIM,
        dim_head=HEAD_DIM,
        heads=HEADS,
        out_dim=DIM,
        context_pre_only=False,
        bias=True,
        processor=diffusers.models.attention_processor.JointAttnProcessor2_0(),
        qk_norm="rms_norm",
        eps=1e-6,
    )
    torch_model.eval()
    state = torch_model.state_dict()

    spatial = torch.randn((1, spatial_seq_len, DIM))
    prompt = torch.randn((1, prompt_seq_len, DIM))
    with torch.no_grad():
        torch_spatial, torch_prompt = torch_model(hidden_states=spatial, encoder_hidden_states=prompt)

    def run_tt(precision, kv_dtype):
        tt_model = SD35JointAttention(
            query_dim=DIM,
            head_dim=HEAD_DIM,
            heads=HEADS,
            out_dim=DIM,
            bias=True,
            out_bias=True,
            context_pre_only=False,
            eps=1e-6,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            padding_config=None,
            sdpa_precision=precision,
            sdpa_kv_dtype=kv_dtype,
        )
        tt_model.load_torch_state_dict({k: v.clone() for k, v in state.items()})

        tt_spatial = bf16_tensor(spatial.unsqueeze(0), device=mesh_device, mesh_axis=sp_axis, shard_dim=2)
        tt_prompt = bf16_tensor(prompt.unsqueeze(0), device=mesh_device)
        tt_spatial_out, tt_prompt_out = tt_model(tt_spatial, tt_prompt, N=spatial_seq_len)
        return {
            "spatial": tensor.to_torch(tt_spatial_out, mesh_axes=[..., sp_axis, tp_axis]).reshape(-1, DIM).float(),
            "prompt": tensor.to_torch(tt_prompt_out, mesh_axes=[..., None, tp_axis]).reshape(-1, DIM).float(),
        }

    gate_vs_legacy(
        record_property,
        f"sd35.{tuple(mesh_device.shape)}.s{spatial_seq_len}.p{prompt_seq_len}",
        {"spatial": torch_spatial.reshape(-1, DIM), "prompt": torch_prompt.reshape(-1, DIM)},
        run_variants(run_tt),
        BOUNDS,
    )


# ---------------------------------------------------------------------------------------------
# Motif joint attention (blocks/attention.py, context head scaling)
# ---------------------------------------------------------------------------------------------


@MESHES
@pytest.mark.parametrize(
    ("spatial_seq_len", "prompt_seq_len"),
    [
        pytest.param(1024, 333, id="s1024_p333"),
        pytest.param(1000, 333, id="s1000_p333"),
    ],
)
def test_motif_attention_sdpa_recipes(
    mesh_device, sp_axis, tp_axis, spatial_seq_len, prompt_seq_len, record_property
) -> None:
    torch.manual_seed(0)
    ccl_manager, parallel_config, sp_factor = parallel(mesh_device, sp_axis, tp_axis)
    if sp_factor == 1 and spatial_seq_len % 32:
        pytest.skip("Motif pads the spatial sequence only with sequence parallelism")
    k_chunk_size = MotifTransformer.get_k_chunk_size(sp_factor)

    class ReferenceAttnConfig:
        hidden_dim = DIM
        num_attention_heads = HEADS
        attn_mode = "flash"

    torch_model = MotifAttentionReference(ReferenceAttnConfig())
    with torch.no_grad():
        torch_model.q_scale.uniform_(0.5, 1.5)  # exercise the context head scaling
    torch_model.eval()

    state = dict(torch_model.state_dict())
    convert_motif_attention_state(
        state,
        x_weight=torch.eye(DIM),
        x_bias=torch.zeros([DIM]),
        c_weight=torch.eye(DIM),
        c_bias=torch.zeros([DIM]),
        is_last_block=False,
    )

    spatial = torch.randn((1, spatial_seq_len, DIM))
    prompt = torch.randn((1, prompt_seq_len, DIM))
    with torch.no_grad():
        torch_spatial, torch_prompt = torch_model.forward(spatial, prompt)

    def run_tt(precision, kv_dtype):
        # As in MotifTransformer's TransformerBlock.
        tt_model = Attention(
            query_dim=DIM,
            head_dim=HEAD_DIM,
            heads=HEADS,
            out_dim=DIM,
            added_kv_proj_dim=DIM,
            context_pre_only=False,
            context_head_scaling=True,
            eps=1e-6,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            padding_config=None,
            sdpa_precision=precision,
            sdpa_kv_dtype=kv_dtype,
        )
        tt_model.load_torch_state_dict({k: v.clone() for k, v in state.items()})

        spatial_padded = Attention.pad_spatial_sequence(spatial, sp_factor=sp_factor, k_chunk_size=k_chunk_size)
        tt_spatial = bf16_tensor(spatial_padded, device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)
        tt_prompt = bf16_tensor(prompt, device=mesh_device)
        tt_spatial_out, tt_prompt_out = tt_model.forward(
            spatial=tt_spatial, prompt=tt_prompt, spatial_sequence_length=spatial_seq_len
        )
        return {
            "spatial": tensor.to_torch(tt_spatial_out, mesh_axes=[None, sp_axis, tp_axis])[:, :spatial_seq_len]
            .reshape(-1, DIM)
            .float(),
            "prompt": tensor.to_torch(tt_prompt_out, mesh_axes=[None, None, tp_axis]).reshape(-1, DIM).float(),
        }

    gate_vs_legacy(
        record_property,
        f"motif.{tuple(mesh_device.shape)}.s{spatial_seq_len}.p{prompt_seq_len}",
        {"spatial": torch_spatial.reshape(-1, DIM), "prompt": torch_prompt.reshape(-1, DIM)},
        run_variants(run_tt),
        BOUNDS,
    )
