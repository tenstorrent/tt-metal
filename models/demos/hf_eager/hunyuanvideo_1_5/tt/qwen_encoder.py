# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device Qwen2.5-VL text encoder for HunyuanVideo-1.5.

Wraps the tt_dit Qwen2.5-VL port so it can replace the CPU `Qwen2_5_VLTextModel`
(`pipe.text_encoder`) in the diffusers pipeline. HunyuanVideo consumes
`hidden_states[-3]` (num_hidden_layers_to_skip=2) from the encoder; the tt port
returns exactly that layer via `hidden_layers_to_skip=2` (validated PCC ~0.995).
"""
from __future__ import annotations

import types

import ttnn
from models.tt_dit.encoders.qwen25vl.model_qwen25vl import Qwen25VlTextEncoder
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import tensor as _tensor

# Shared holder for the on-device Qwen submesh (set by the test's mesh_device
# fixture, read by the gen path) -- same pattern/rationale as
# vae_decoder.HY_VAE_SUBMESH (pytest imports conftest under a private name).
HY_QWEN_SUBMESH = None

_HIDDEN_LAYERS_TO_SKIP = 2  # HunyuanVideo mllm embed = hidden_states[-(2+1)] = [-3]


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
        # _get_mllm_prompt_embeds reads only hidden_states[-3]; return a 3-list so [-3]==emb.
        return types.SimpleNamespace(hidden_states=[emb, emb, emb])
