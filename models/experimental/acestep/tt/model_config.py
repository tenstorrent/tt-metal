# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ACE-Step v1.5 model config + TT model factory.

Canonical flow (mirrors Phi-4 `from_pretrained` and BGE-M3 `ModelArgs.load_model`):

    args     = AceStepModelConfig.from_hf(mesh_device)         # dims from HF config
    pipeline = create_tt_pipeline(args, mesh_device)           # DiT + VAE, full text-to-music path

`create_tt_pipeline` (in tt/pipeline.py) is the single public entry point. It assembles the
DiT denoiser (via the internal `_build_dit_model` here) plus the Oobleck VAE decoder. The DiT
builder reuses the validated per-module configs and the `reference/weight_utils` loader; encoders
(lyric/timbre/text) are exposed via `build_condition_encoder` for the full pipeline.

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


def _build_dit_model(args: AceStepModelConfig, mesh_device) -> AceStepDiTModel:
    """Build the TT AceStepDiTModel (flow-matching denoiser) from the checkpoint.

    Internal DiT builder for `create_tt_pipeline`. Returns an AceStepDiTModel ready for
    forward(hidden, context_latents, t, t_r, cos, sin, encoder_hidden_states, sliding_mask).
    Weights load lazily on first forward.
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


def _text_encoder_layer_config(rl, args, device, compute_kernel_config=None):
    """AceStepEncoderLayerConfig for a Qwen3 text-encoder layer (full attention, causal via mask).

    compute_kernel_config: optional matmul fidelity override (None keeps the attention/MLP HiFi2
    defaults; the LM planner passes HiFi4+fp32-acc since its fp32 reference rewards higher precision).
    """
    a = rl.self_attn
    dt = args.weight_dtype
    return AceStepEncoderLayerConfig(
        compute_kernel_config=compute_kernel_config,
        input_layernorm_weight=lazy_norm(rl.input_layernorm.weight, device, dt),
        post_attention_layernorm_weight=lazy_norm(rl.post_attention_layernorm.weight, device, dt),
        wq=lazy_wT(a.q_proj.weight, device, dt),
        wk=lazy_wT(a.k_proj.weight, device, dt),
        wv=lazy_wT(a.v_proj.weight, device, dt),
        wo=lazy_wT(a.o_proj.weight, device, dt),
        q_norm_weight=lazy_norm(a.q_norm.weight, device, dt),
        k_norm_weight=lazy_norm(a.k_norm.weight, device, dt),
        w1=lazy_wT(rl.mlp.gate_proj.weight, device, dt),
        w2=lazy_wT(rl.mlp.down_proj.weight, device, dt),
        w3=lazy_wT(rl.mlp.up_proj.weight, device, dt),
        n_heads=a.config.num_attention_heads,
        n_kv_heads=a.config.num_key_value_heads,
        head_dim=a.head_dim,
        eps=rl.input_layernorm.variance_epsilon,
        sliding_window=None,
    )


def _build_qwen3_encoder(mesh_device, subdir: str, *, dtype=None, compute_kernel_config=None):
    """Build a TT AceStepTextEncoder from a causal Qwen3Model checkpoint in the pipeline bundle.

    Both the ACE-Step text encoder (Qwen3-Embedding-0.6B) and the 5Hz LM planner
    (acestep-5Hz-lm-1.7B) are base causal Qwen3Model with the SAME layer structure (self_attn
    q/k/v/o + q/k-norm, SwiGLU MLP, pre-norms). They differ only in dims (hidden 1024 vs 2048),
    vocab, and layer count — all read from the checkpoint — so one builder covers both. Reuses
    AceStepEncoderLayer + RMSNorm1D; causal via an additive mask; vocab embed on host.
    """
    import ttnn as _ttnn
    from transformers import AutoModel
    from models.experimental.acestep.reference.weight_utils import pipeline_dir
    from models.experimental.acestep.tt.text_encoder import AceStepTextEncoder, AceStepTextEncoderConfig

    dt = dtype or _ttnn.bfloat16
    hf_te = AutoModel.from_pretrained(str(pipeline_dir() / subdir), dtype=torch.float32).eval()

    class _A:  # tiny arg carrier so the layer helper can read weight_dtype
        weight_dtype = dt

    args = _A()
    cfg = AceStepTextEncoderConfig(
        embed_tokens=hf_te.embed_tokens.weight.detach(),
        norm_weight=lazy_norm(hf_te.norm.weight, mesh_device, dt),
        layer_configs=[
            _text_encoder_layer_config(rl, args, mesh_device, compute_kernel_config=compute_kernel_config)
            for rl in hf_te.layers
        ],
        hidden_size=hf_te.config.hidden_size,
        eps=hf_te.config.rms_norm_eps,
        compute_kernel_config=compute_kernel_config,
    )
    return AceStepTextEncoder.from_config(cfg), hf_te


def build_text_encoder(mesh_device, *, dtype=None):
    """Build the TT text encoder (Qwen3-Embedding-0.6B): prompt tokens -> text_hidden_states."""
    return _build_qwen3_encoder(mesh_device, "Qwen3-Embedding-0.6B", dtype=dtype)


def build_lm_planner(mesh_device, *, dtype=None):
    """Build the TT 5Hz LM planner base (acestep-5Hz-lm-1.7B): a causal Qwen3Model, hidden 2048.

    Reuses the exact same Qwen3-encoder path as the text encoder (same layer structure). Returns
    the last_hidden_state model; the tied-embedding LM head (logits) is a separate projection.

    The LM has massive activations (absmax ~200+) that bf16 mis-represents; unlike the DiT (whose
    reference is bf16-regime), the LM reference is fp32 last_hidden_state, so HiFi4 math fidelity +
    fp32 accumulate move TOWARD the reference. Applied to every LM layer's matmuls.
    """
    import ttnn as _ttnn

    ckc = _ttnn.WormholeComputeKernelConfig(
        math_fidelity=_ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    return _build_qwen3_encoder(mesh_device, "acestep-5Hz-lm-1.7B", dtype=dtype, compute_kernel_config=ckc)


def build_vae_decoder(mesh_device, *, dtype=None):
    """Build the TT Oobleck VAE decoder (latents -> 48kHz waveform) from the genuine checkpoint.

    Reuses the TTTv2-primitive OobleckDecoder and loads the diffusers AutoencoderOobleck weights
    (effective weight-norm-folded weights). fp32 audio path (matches vocoder_ltx).
    """
    import ttnn as _ttnn
    from diffusers import AutoencoderOobleck
    from models.experimental.acestep.reference.weight_utils import vae_dir
    from models.experimental.acestep.tt.vae_decoder import OobleckDecoder, OobleckVAEConfig
    from models.experimental.acestep.tt.vae_conv_config import apply_vae_conv3d_config, vae_default_dtype

    # VAE runs bf16 by default (no fp32 on the audio path) — external, env-overridable config.
    dtype = dtype or vae_default_dtype()
    # Apply the external, tuned Conv3d blockings for (arch, dtype) before building the convs (no-op
    # when no preset exists -> tt_dit fallback). Non-hardcoded: keyed by arch+dtype, sweep-derived.
    apply_vae_conv3d_config(mesh_device, dtype)
    vae = AutoencoderOobleck.from_pretrained(vae_dir()).eval()
    cfg = OobleckVAEConfig.from_diffusers(vae.config)
    dec = OobleckDecoder(cfg, mesh_device=mesh_device, dtype=dtype)
    dec.load_torch_state_dict(_effective_vae_decoder_state(vae.decoder))
    return dec, cfg


def _fold_weight_norm(ref_decoder) -> None:
    """Materialize + strip weight_norm on every conv, in-place (standard torch API).

    diffusers stores weight_norm as `weight_g`/`weight_v` and computes `.weight` lazily via a
    parametrization — so `.weight` is a META tensor until the first forward. tt_dit's conv modules
    expect a plain `weight`, so we fold weight_norm the idiomatic way (`remove_weight_norm`), which
    materializes `.weight` to a real tensor and removes the g/v buffers. Same approach the tt_dit
    LTX vocoder path assumes (its diffusers weights arrive already weight-norm-free).
    """
    import torch.nn as _nn
    from torch.nn.utils import remove_weight_norm

    for mod in ref_decoder.modules():
        if isinstance(mod, (_nn.Conv1d, _nn.ConvTranspose1d)) and hasattr(mod, "weight_g"):
            remove_weight_norm(mod)


def _effective_vae_decoder_state(ref_decoder) -> dict:
    """State dict of EFFECTIVE (weight-norm folded) VAE decoder weights, keyed to the TT tree."""
    import torch.nn as _nn
    from diffusers.models.autoencoders.autoencoder_oobleck import Snake1d as _Snake1d

    _fold_weight_norm(ref_decoder)
    state: dict = {}
    for name, mod in ref_decoder.named_modules():
        if isinstance(mod, (_nn.Conv1d, _nn.ConvTranspose1d)):
            state[f"{name}.weight"] = mod.weight.detach()
            if mod.bias is not None:
                state[f"{name}.bias"] = mod.bias.detach()
        elif isinstance(mod, _Snake1d):
            state[f"{name}.alpha"] = mod.alpha.detach().reshape(-1)
            state[f"{name}.beta"] = mod.beta.detach().reshape(-1)
    return state
