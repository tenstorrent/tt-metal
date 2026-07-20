# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device Qwen2.5-VL text encoder for HunyuanVideo-1.5.

Wraps the tt_dit Qwen2.5-VL port so it can replace the CPU `Qwen2_5_VLTextModel`
(`pipe.text_encoder`) in the diffusers pipeline. HunyuanVideo consumes
`hidden_states[-3]` (num_hidden_layers_to_skip=2) from the encoder; the tt port
returns exactly that layer via `hidden_layers_to_skip=2` (validated PCC ~0.995).
"""
from __future__ import annotations

import os
import types

import ttnn
from models.tt_dit.encoders.qwen25vl.model_qwen25vl import Qwen25VlTextEncoder, _apply_rope
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import tensor as _tensor

# Shared holder for the on-device Qwen submesh (set by the test's mesh_device
# fixture, read by the gen path) -- same pattern/rationale as
# vae_decoder.HY_VAE_SUBMESH (pytest imports conftest under a private name).
HY_QWEN_SUBMESH = None

_HIDDEN_LAYERS_TO_SKIP = 2  # HunyuanVideo mllm embed = hidden_states[-(2+1)] = [-3]


def _eager_attn_fp32_forward(self, x, *, attention_bias, pos_embeds):
    """fp32 eager attention (explicit q@kT -> softmax -> @v) replacing the flash
    SDPA. Fixes the HunyuanVideo conditioning-fidelity gap: the bf16 attention core
    (flash or bf16-eager) accumulates ~1.3%/layer error over 26 layers -> ~0.99
    conditioning that blurs the diffusion output. Doing the core in fp32 recovers it
    to ~0.9998 (see the note at the bottom of this file). Text-encode runs once per
    generation, so the fp32/explicit cost is acceptable. Mirrors
    Qwen25VlAttention.forward exactly otherwise (qkv/head-split/rope/o_proj)."""
    x = self.qkv_proj.forward(x)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        ttnn.unsqueeze(x, 1),
        num_heads=self._num_local_heads,
        num_kv_heads=self._num_local_kv_heads,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cos, sin = pos_embeds
    q = _apply_rope(q, cos, sin)
    k = _apply_rope(k, cos, sin)
    hq, hkv = self._num_local_heads, self._num_local_kv_heads
    if hq != hkv:  # GQA: broadcast kv heads to q heads
        k = ttnn.repeat_interleave(k, hq // hkv, dim=1)
        v = ttnn.repeat_interleave(v, hq // hkv, dim=1)
    q = ttnn.typecast(q, ttnn.float32)
    k = ttnn.typecast(k, ttnn.float32)
    v = ttnn.typecast(v, ttnn.float32)
    scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = ttnn.multiply(
        ttnn.matmul(q, ttnn.permute(k, (0, 1, 3, 2)), compute_kernel_config=self._sdpa_compute_kernel_config), scale
    )
    if attention_bias is not None:
        scores = ttnn.add(scores, ttnn.typecast(attention_bias, ttnn.float32))
    attn = ttnn.softmax(scores, dim=-1)
    out = ttnn.matmul(attn, v, compute_kernel_config=self._sdpa_compute_kernel_config)
    out = ttnn.typecast(out, ttnn.bfloat16)
    x = ttnn.transformer.concatenate_heads(out)
    if self._tp_axis is not None:
        x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)
    x = self.o_proj.forward(x)
    if self._tp_axis is not None:
        x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)
    return x


def build_tt_qwen_encoder(text_encoder, device) -> Qwen25VlTextEncoder:
    """Build the tt_dit Qwen text-tower encoder from a torch `Qwen2_5_VLTextModel`,
    tensor-parallel across `device` (a 1xN submesh)."""
    cfg = text_encoder.config
    rope_params = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
    rope_theta = getattr(cfg, "rope_theta", None) or rope_params.get("rope_theta")
    tp = device.get_num_devices()
    model = Qwen25VlTextEncoder(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        hidden_act=cfg.hidden_act,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=rope_theta,
        mrope_section=rope_params["mrope_section"],
        device=device,
        parallel_config=EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=tp, mesh_axis=1)),
        ccl_manager=CCLManager(device, num_links=1, topology=ttnn.Topology.Linear) if tp > 1 else None,
    )
    model.load_torch_state_dict(text_encoder.state_dict())
    # Fidelity fix: use fp32 eager attention instead of the bf16 flash SDPA. The
    # bf16 attention core blurs HunyuanVideo's conditioning (~0.99 PCC); fp32 eager
    # restores ~0.9998. See _eager_attn_fp32_forward and the note below.
    for layer in model.layers:
        layer.self_attn.forward = types.MethodType(_eager_attn_fp32_forward, layer.self_attn)
    return model


class TTQwenTextEncoderAdapter:
    """Drop-in for `pipe.text_encoder` (Qwen2_5_VLTextModel) that runs the mllm
    text encode on device. Only `_get_mllm_prompt_embeds` calls it, and only reads
    `.hidden_states[-3]`, so `__call__` returns a namespace with a 3-long
    hidden_states list whose [-3] element is the tt encoder output."""

    def __init__(self, real_text_encoder, device, *, dtype=ttnn.bfloat16):
        self.__dict__["_real"] = real_text_encoder
        self.__dict__["_device"] = device
        self.__dict__["_tt"] = build_tt_qwen_encoder(real_text_encoder, device)

    def __getattr__(self, k):
        return getattr(self.__dict__["_real"], k)

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=True, **kw):
        dev = self.__dict__["_device"]
        tt = self.__dict__["_tt"]
        ids = input_ids.to("cpu")
        mask = attention_mask.to("cpu") if attention_mask is not None else None
        cos, sin = tt.create_rope_tensors(ids.shape[0], ids.shape[1], mask)
        tt_ids = _tensor.from_torch(ids, device=dev, dtype=ttnn.uint32)
        tt_mask = _tensor.from_torch(mask, device=dev) if mask is not None else None
        tt_cos = _tensor.from_torch(cos, device=dev)
        tt_sin = _tensor.from_torch(sin, device=dev)
        hs = tt.forward(
            tt_ids, attention_mask=tt_mask, pos_embeds=(tt_cos, tt_sin), hidden_layers_to_skip=_HIDDEN_LAYERS_TO_SKIP
        )
        emb = ttnn.to_torch(ttnn.get_device_tensors(hs[-1])[0]).to(self.__dict__["_real"].dtype)
        # Zero the PADDING-token embeds (WAN's strategy, models/tt_dit/pipelines/wan).
        # The tt Qwen port reproduces the ~valid tokens well (PCC 0.9998) but its
        # PADDING embeds are garbage (PCC ~0.66, rel-L2 ~0.94). The DiT's fused joint
        # SDPA can't mask padding, and _trim_to_valid trims a mixed-length CFG batch
        # to the LONGEST row -- so a shorter row's (e.g. uncond) garbage padding leaks
        # into attention and, amplified by CFG, blurs the video. Zeroing padding makes
        # those leaked tokens neutral instead of garbage. Off via HY_QWEN_ZERO_PAD=0.
        if mask is not None and os.environ.get("HY_QWEN_ZERO_PAD", "1") == "1":
            emb = emb * mask[..., : emb.shape[1]].unsqueeze(-1).to(emb.dtype)
        # _get_mllm_prompt_embeds reads only hidden_states[-3]; return a 3-list so [-3]==emb.
        return types.SimpleNamespace(hidden_states=[emb, emb, emb])


# ---------------------------------------------------------------------------
# NOTE / TODO (Qwen text-encode fidelity) — for a proper upstream fix
# ---------------------------------------------------------------------------
# Symptom: with the stock tt_dit Qwen port, on-device text-encode produced a
# noticeably blurry video vs CPU text-encode (same DiT+VAE otherwise).
#
# Investigation (repro scripts kept in the bring-up scratchpad):
#   - NOT bf16 rounding:  CPU-bf16 vs CPU-fp32 = 0.99994.
#   - NOT tensor parallel: TP=1 == TP=4 (0.9906).
#   - NOT the attn mask:   bf4 == bf16 == finite (0.98731).
#   - NOT the MLP:         isolated (matched input) = 0.99999.
#   - NOT qkv/head split:  pre-RoPE q/k/v = 1.0 / 1.0 / 0.99999.
#   => The gap is the ATTENTION CORE at bf16 (RoPE + SDPA), ~1.3%/layer,
#      accumulating over 26 layers to ~0.99 on the hidden_states[-3] the DiT
#      consumes -- enough to blur the diffusion output.
#
# What does / doesn't fix it (measured on the 13 valid conditioning tokens):
#   flash SDPA (bf16 core, HiFi4+fp32 acc): 0.99145   (blurs)
#   bf16 eager attention:                   0.97476   (worse)
#   fp32 eager attention (THIS):            0.99980   (fixed)
# HiFi4 on the Linears barely moved it (0.9915->0.9934). ttnn.transformer.
# scaled_dot_product_attention REJECTS fp32 inputs, so fp32 must be done via
# explicit eager attention (no SDPA) -- what _eager_attn_fp32_forward does.
#
# This module applies the fp32-eager workaround locally (monkeypatch on each
# layer's self_attn). PROPER FIX belongs upstream in models/tt_dit/encoders/
# qwen25vl: either (a) an fp32 attention-core option, or (b) improve the bf16
# flash-SDPA path's numerics for this deep-network / mid-layer-readout use.
# Text-encode runs ONCE per generation, so the fp32/explicit cost here is fine.
#
# UPDATE (validated end-to-end): fp32-eager brings the VALID-token embeds to
# PCC 0.99984 / rel-L2 1.8% / norm-ratio 0.997 / linear-fit s=1,c=0 vs CPU-fp32
# (at TP=4, the gen config) -- a genuine correction over bf16 flash (~0.99). BUT a
# full 13-frame gen with this fix STILL looked soft vs CPU text-encode. So the
# residual video degradation is NOT the valid-token embed accuracy (now matched).
# Prime remaining suspect: the ~987 PADDING tokens (post-crop mllm stream is 13
# valid + ~987 pad) -- the DiT's fused joint-SDPA can't mask padding (see the
# joint-SDPA commit note), and _trim_to_valid only trims to a tile-multiple, so a
# few padding embeds leak into DiT attention; tt vs CPU padding embeds can differ.
# NEXT: compare tt-vs-CPU embeds over the padding region and audit _trim_to_valid /
# the DiT mllm padding handling. (Diffusion sensitivity to the residual is also
# possible.) Keep text-encode on CPU for production until this is closed.
