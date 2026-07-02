# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ACE-Step v1.5 model config + TT model factory.

Canonical flow (mirrors Phi-4 `from_pretrained` and BGE-M3 `ModelArgs.load_model`):

    args  = AceStepModelConfig.from_hf(mesh_device)      # dims from HF config
    model = create_tt_model(args, mesh_device)           # builds full TT DiT from real weights

`create_tt_model` returns an `AceStepDiTModel` (the flow-matching denoiser — the core generative
compute) wired from the genuine checkpoint. It reuses the validated per-module configs and the
`reference/weight_utils` loader. Encoders (lyric/timbre/text) are exposed via `build_condition_encoder`
for the full pipeline.

Weights always load from the genuine `model.safetensors` checkpoint (no dummy-weight path).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

import ttnn

from models.experimental.acestep.reference.hf_reference import load_config, load_modeling_module
from models.experimental.acestep.reference.weight_utils import load_module_weights
from models.common.modules.lazy_weight import LazyWeight
from models.experimental.acestep.tt.condition_encoder import (
    AceStepConditionEncoder,
    AceStepConditionEncoderConfig,
)
from models.experimental.acestep.tt.dit_layer import AceStepDiTLayerConfig
from models.experimental.acestep.tt.dit_model import AceStepDiTModel, AceStepDiTModelConfig
from models.experimental.acestep.tt.dit_output import DiTOutputConfig
from models.experimental.acestep.tt.dit_stack import AceStepDiTStackConfig
from models.experimental.acestep.tt.encoder_layer import AceStepEncoderLayerConfig
from models.experimental.acestep.tt.lyric_encoder import AceStepLyricEncoderConfig
from models.experimental.acestep.tt.patch_embed import PatchEmbedConfig
from models.experimental.acestep.tt.timestep_embedding import TimestepEmbeddingConfig

HF_MODEL_ID = "ACE-Step/acestep-v15-base"
TIME_IN_CHANNELS = 256  # TimestepEmbedding sinusoidal input dim (fixed by the reference)


# =============================================================================
# Weight extraction helpers (single source of truth; tests import these)
# =============================================================================


def lazy_wT(weight: torch.Tensor, device, dtype: ttnn.DataType = ttnn.bfloat16) -> LazyWeight:
    """HF Linear weight [out,in] -> [in,out] for ttnn.linear, wrapped as LazyWeight (TILE)."""
    return LazyWeight(
        source=weight.detach().clone().transpose(-1, -2).contiguous(),
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def lazy_norm(weight: torch.Tensor, device, dtype: ttnn.DataType = ttnn.bfloat16) -> LazyWeight:
    """RMSNorm weight (ROW_MAJOR)."""
    return LazyWeight(
        source=weight.detach().clone(),
        device=device,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def lazy_bias(bias: torch.Tensor, device, dtype: ttnn.DataType = ttnn.bfloat16) -> LazyWeight:
    """Bias [out] -> [1,out] (TILE)."""
    return LazyWeight(
        source=bias.detach().clone().reshape(1, -1).contiguous(),
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def lazy_raw(
    tensor: torch.Tensor, device, dtype: ttnn.DataType = ttnn.bfloat16, layout: ttnn.Layout = ttnn.TILE_LAYOUT
) -> LazyWeight:
    return LazyWeight(
        source=tensor.detach().clone(), device=device, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


# =============================================================================
# Model config
# =============================================================================


@dataclass
class AceStepModelConfig:
    """ACE-Step v1.5 dims + build settings. All architecture fields sourced from the HF config."""

    # Architecture (HF config.json).
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    sliding_window: int = 128
    in_channels: int = 192
    audio_acoustic_hidden_dim: int = 64
    patch_size: int = 2
    text_hidden_dim: int = 1024
    timbre_hidden_dim: int = 64
    pool_window_size: int = 5
    num_lyric_encoder_hidden_layers: int = 8
    num_timbre_encoder_hidden_layers: int = 4

    # Build settings.
    hf_model_id: str = HF_MODEL_ID
    weight_dtype: ttnn.DataType = ttnn.bfloat16
    layer_types: list = field(default_factory=list)  # per-layer full/sliding, from HF config

    @classmethod
    def from_hf(cls, *, num_hidden_layers: int | None = None) -> "AceStepModelConfig":
        """Build config from the real HF model config (dims never hard-coded blindly)."""
        hf = load_config()
        n = num_hidden_layers if num_hidden_layers is not None else hf.num_hidden_layers
        return cls(
            hidden_size=hf.hidden_size,
            intermediate_size=hf.intermediate_size,
            num_hidden_layers=n,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=hf.num_key_value_heads,
            head_dim=hf.head_dim,
            rms_norm_eps=hf.rms_norm_eps,
            rope_theta=hf.rope_theta,
            sliding_window=hf.sliding_window,
            in_channels=hf.in_channels,
            audio_acoustic_hidden_dim=hf.audio_acoustic_hidden_dim,
            patch_size=hf.patch_size,
            text_hidden_dim=hf.text_hidden_dim,
            timbre_hidden_dim=hf.timbre_hidden_dim,
            pool_window_size=hf.pool_window_size,
            num_lyric_encoder_hidden_layers=hf.num_lyric_encoder_hidden_layers,
            num_timbre_encoder_hidden_layers=hf.num_timbre_encoder_hidden_layers,
            hf_model_id=HF_MODEL_ID,
            layer_types=list(hf.layer_types[:n]),
        )

    def sliding_for(self, attn_type: str) -> int | None:
        return self.sliding_window if attn_type == "sliding_attention" else None


# =============================================================================
# Reference-module loaders (real or dummy weights)
# =============================================================================


def _load_reference_dit(args: AceStepModelConfig):
    """Instantiate the HF AceStepDiTModel and populate genuine checkpoint weights."""
    m = load_modeling_module()
    hf = load_config()
    hf._attn_implementation = "eager"
    hf.num_hidden_layers = args.num_hidden_layers
    dit = m.AceStepDiTModel(hf).eval()
    load_module_weights(dit, "decoder.", allow_extra=True)
    return m, dit


def _load_reference_condition_encoder(args: AceStepModelConfig):
    m = load_modeling_module()
    hf = load_config()
    hf._attn_implementation = "eager"
    ce = m.AceStepConditionEncoder(hf).eval()
    load_module_weights(ce, "encoder.")
    return m, ce


# =============================================================================
# Config builders (per module) — reuse validated module configs
# =============================================================================


def _dit_layer_config(rl, attn_type, args, device) -> AceStepDiTLayerConfig:
    a, c = rl.self_attn, rl.cross_attn
    return AceStepDiTLayerConfig(
        scale_shift_table=lazy_raw(rl.scale_shift_table, device, args.weight_dtype),
        self_attn_norm_weight=lazy_norm(rl.self_attn_norm.weight, device, args.weight_dtype),
        mlp_norm_weight=lazy_norm(rl.mlp_norm.weight, device, args.weight_dtype),
        cross_attn_norm_weight=lazy_norm(rl.cross_attn_norm.weight, device, args.weight_dtype),
        wq=lazy_wT(a.q_proj.weight, device, args.weight_dtype),
        wk=lazy_wT(a.k_proj.weight, device, args.weight_dtype),
        wv=lazy_wT(a.v_proj.weight, device, args.weight_dtype),
        wo=lazy_wT(a.o_proj.weight, device, args.weight_dtype),
        q_norm_weight=lazy_norm(a.q_norm.weight, device, args.weight_dtype),
        k_norm_weight=lazy_norm(a.k_norm.weight, device, args.weight_dtype),
        c_wq=lazy_wT(c.q_proj.weight, device, args.weight_dtype),
        c_wk=lazy_wT(c.k_proj.weight, device, args.weight_dtype),
        c_wv=lazy_wT(c.v_proj.weight, device, args.weight_dtype),
        c_wo=lazy_wT(c.o_proj.weight, device, args.weight_dtype),
        c_q_norm_weight=lazy_norm(c.q_norm.weight, device, args.weight_dtype),
        c_k_norm_weight=lazy_norm(c.k_norm.weight, device, args.weight_dtype),
        w1=lazy_wT(rl.mlp.gate_proj.weight, device, args.weight_dtype),
        w2=lazy_wT(rl.mlp.down_proj.weight, device, args.weight_dtype),
        w3=lazy_wT(rl.mlp.up_proj.weight, device, args.weight_dtype),
        n_heads=args.num_attention_heads,
        n_kv_heads=args.num_key_value_heads,
        head_dim=args.head_dim,
        dim=args.hidden_size,
        eps=args.rms_norm_eps,
        sliding_window=args.sliding_for(attn_type),
        use_cross_attention=True,
    )


def _timestep_config(te, args, device) -> TimestepEmbeddingConfig:
    return TimestepEmbeddingConfig(
        linear_1_weight=lazy_wT(te.linear_1.weight, device, args.weight_dtype),
        linear_1_bias=lazy_bias(te.linear_1.bias, device, args.weight_dtype),
        linear_2_weight=lazy_wT(te.linear_2.weight, device, args.weight_dtype),
        linear_2_bias=lazy_bias(te.linear_2.bias, device, args.weight_dtype),
        time_proj_weight=lazy_wT(te.time_proj.weight, device, args.weight_dtype),
        time_proj_bias=lazy_bias(te.time_proj.bias, device, args.weight_dtype),
        in_channels=TIME_IN_CHANNELS,
        time_embed_dim=args.hidden_size,
    )


def _encoder_layer_config(rl, attn_type, args, device) -> AceStepEncoderLayerConfig:
    a = rl.self_attn
    return AceStepEncoderLayerConfig(
        input_layernorm_weight=lazy_norm(rl.input_layernorm.weight, device, args.weight_dtype),
        post_attention_layernorm_weight=lazy_norm(rl.post_attention_layernorm.weight, device, args.weight_dtype),
        wq=lazy_wT(a.q_proj.weight, device, args.weight_dtype),
        wk=lazy_wT(a.k_proj.weight, device, args.weight_dtype),
        wv=lazy_wT(a.v_proj.weight, device, args.weight_dtype),
        wo=lazy_wT(a.o_proj.weight, device, args.weight_dtype),
        q_norm_weight=lazy_norm(a.q_norm.weight, device, args.weight_dtype),
        k_norm_weight=lazy_norm(a.k_norm.weight, device, args.weight_dtype),
        w1=lazy_wT(rl.mlp.gate_proj.weight, device, args.weight_dtype),
        w2=lazy_wT(rl.mlp.down_proj.weight, device, args.weight_dtype),
        w3=lazy_wT(rl.mlp.up_proj.weight, device, args.weight_dtype),
        n_heads=args.num_attention_heads,
        n_kv_heads=args.num_key_value_heads,
        head_dim=args.head_dim,
        eps=args.rms_norm_eps,
        sliding_window=args.sliding_for(attn_type),
    )


def _lyric_encoder_config(enc, args, device) -> AceStepLyricEncoderConfig:
    types = [rl.attention_type for rl in enc.layers]
    return AceStepLyricEncoderConfig(
        embed_weight=lazy_wT(enc.embed_tokens.weight, device, args.weight_dtype),
        embed_bias=lazy_bias(enc.embed_tokens.bias, device, args.weight_dtype),
        norm_weight=lazy_norm(enc.norm.weight, device, args.weight_dtype),
        layer_configs=[_encoder_layer_config(rl, at, args, device) for rl, at in zip(enc.layers, types)],
        layer_types=types,
        eps=args.rms_norm_eps,
    )


# =============================================================================
# Factory functions
# =============================================================================


def create_tt_model(args: AceStepModelConfig, mesh_device) -> AceStepDiTModel:
    """Build the TT AceStepDiTModel (flow-matching denoiser) from the checkpoint.

    Returns an AceStepDiTModel ready for forward(hidden, context_latents, t, t_r, cos, sin,
    encoder_hidden_states, sliding_mask). Weights load lazily on first forward.
    """
    m, dit = _load_reference_dit(args)
    device = mesh_device
    dt = args.weight_dtype

    layer_types = [rl.attention_type for rl in dit.layers]

    # proj_in Conv1d(in,out,k=stride=p) folded to patchify+linear [in*p, out].
    conv = dit.proj_in[1]
    oc, ic, kk = conv.weight.shape
    proj_in_w = conv.weight.reshape(oc, ic * kk).transpose(0, 1).contiguous()

    # proj_out ConvTranspose1d(in,out,k=stride=p) folded to linear [in, out*p] (k,out order).
    ct = dit.proj_out[1]
    inp, outp, k = ct.weight.shape
    proj_out_w = ct.weight.permute(2, 1, 0).reshape(k * outp, inp).transpose(0, 1).contiguous()

    model = AceStepDiTModel(
        AceStepDiTModelConfig(
            time_embed=_timestep_config(dit.time_embed, args, device),
            time_embed_r=_timestep_config(dit.time_embed_r, args, device),
            patch_embed=PatchEmbedConfig(
                weight=lazy_raw(proj_in_w, device, dt),
                bias=lazy_bias(conv.bias, device, dt),
                in_channels=args.in_channels,
                out_channels=args.hidden_size,
                patch_size=args.patch_size,
            ),
            condition_embedder_weight=lazy_wT(dit.condition_embedder.weight, device, dt),
            condition_embedder_bias=lazy_bias(dit.condition_embedder.bias, device, dt),
            stack=AceStepDiTStackConfig(
                layer_configs=[_dit_layer_config(rl, at, args, device) for rl, at in zip(dit.layers, layer_types)],
                layer_types=layer_types,
            ),
            output=DiTOutputConfig(
                scale_shift_table=lazy_raw(dit.scale_shift_table, device, dt),
                norm_out_weight=lazy_norm(dit.norm_out.weight, device, dt),
                proj_out_weight=lazy_raw(proj_out_w, device, dt),
                proj_out_bias=lazy_bias(ct.bias, device, dt),
                dim=args.hidden_size,
                out_channels=args.audio_acoustic_hidden_dim,
                patch_size=args.patch_size,
                eps=args.rms_norm_eps,
            ),
            dim=args.hidden_size,
        )
    )
    return model


def build_condition_encoder(args: AceStepModelConfig, mesh_device) -> AceStepConditionEncoder:
    """Build the TT ConditionEncoder (text_projector + lyric + timbre) from the checkpoint."""
    m, ce = _load_reference_condition_encoder(args)
    device = mesh_device
    return AceStepConditionEncoder(
        AceStepConditionEncoderConfig(
            text_projector_weight=lazy_wT(ce.text_projector.weight, device, args.weight_dtype),
            lyric_encoder=_lyric_encoder_config(ce.lyric_encoder, args, device),
            timbre_encoder=_lyric_encoder_config(ce.timbre_encoder, args, device),
        )
    )
