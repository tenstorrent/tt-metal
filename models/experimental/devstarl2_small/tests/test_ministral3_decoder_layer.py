# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# PCC: HF Ministral3DecoderLayer stack vs TtMinistral3DecoderLayer stack decode (Devstral).

from __future__ import annotations

import os
import types

import numpy as np
import pytest
import torch
from loguru import logger
from transformers.cache_utils import DynamicCache
from transformers.models.ministral3.modeling_ministral3 import Ministral3DecoderLayer

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstarl2_small.devstral_utils import apply_fp8_dequantize_compat, resolve_rope_parameters
from models.experimental.devstarl2_small.tt.tt_ministral3_decoder_layer import TtMinistral3DecoderLayer
from models.experimental.devstarl2_small.tt.tt_ministral_rotary_emb import TtMinistral3RotaryEmbedding
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model_config import ModelArgs

apply_fp8_dequantize_compat()

DEVSTRAL_REPO_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"

PCC_TARGET = float(os.environ.get("MINISTRAL3_DECODER_STACK_PCC", "0.99"))
DECODE_STEPS = int(os.environ.get("MINISTRAL3_DECODER_DECODE_STEPS", "5"))
# Profile decode forward only: skip per-step to_torch (UntilizeDeviceOperation in Tracy).
PROFILE_DECODE = os.environ.get("MINISTRAL3_DECODER_PROFILE", "0").strip().lower() in ("1", "true", "yes")


def _text_model_root(multimodal_inner):
    lm = multimodal_inner.language_model
    return lm.model if hasattr(lm, "model") else lm


def _hf_decoder_stack_forward(
    layers,
    rotary,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values: DynamicCache,
) -> torch.Tensor:
    position_embeddings = rotary(hidden_states, position_ids=position_ids)
    cache_position = position_ids[0]
    hidden = hidden_states
    for layer in layers:
        layer_out = layer(
            hidden_states=hidden,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out
    return hidden


@pytest.fixture
def trust_remote_ministral(monkeypatch):
    from models.tt_transformers.tt import model_config as mc

    orig_set = mc.ModelArgs._set_hf_params

    def _set_hf_params_trust(self, checkpoint_dir: str):
        self.trust_remote_code_hf = True
        return orig_set(self, checkpoint_dir)

    monkeypatch.setattr(mc.ModelArgs, "_set_hf_params", _set_hf_params_trust)

    def _get_hf_model_cls_devstral_safe(self):
        from transformers import AutoModelForCausalLM
        from transformers.models.auto.modeling_auto import AutoModelForImageTextToText

        if not self.is_multimodal:
            return AutoModelForCausalLM
        if type(self.hf_config) in AutoModelForImageTextToText._model_mapping:
            return AutoModelForImageTextToText
        raise ValueError(
            f"Test supports multimodal configs in AutoModelForImageTextToText only; got {type(self.hf_config)}"
        )

    monkeypatch.setattr(mc.ModelArgs, "get_hf_model_cls", _get_hf_model_cls_devstral_safe)


_MESH_DEVICE = [{"P150": (1, 1), "BH-QB": (1, 4)}.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))]


@torch.no_grad()
@pytest.mark.timeout(7200)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("mesh_device", _MESH_DEVICE, indirect=True)
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 60000000, "num_command_queues": 1}],
    indirect=True,
)
def test_ministral3_decoder_layer_decode_pcc_devstral_weights(
    mesh_device,
    batch_size,
    monkeypatch,
    trust_remote_ministral,
):
    """Full decoder stack decode PCC vs HF (KV cache filled incrementally; no prefill pass)."""
    monkeypatch.setenv("HF_MODEL", DEVSTRAL_REPO_ID)

    dtype = ttnn.bfloat16
    model_args = ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=512,
        dummy_weights=False,
        use_hf_rope=True,
        cache_hf=True,
    )
    model_args.is_distributed_norm = types.MethodType(
        lambda self, mode: model_args.is_multichip and mode == Mode.DECODE,
        model_args,
    )

    depth = int(model_args.full_model_n_layers)
    n_layers = depth
    logger.info(
        f"Ministral3 decoder stack decode PCC: n_layers={n_layers}/{depth} "
        f"steps={DECODE_STEPS} pcc_required={PCC_TARGET}"
    )

    text_cfg = model_args.hf_config.text_config

    try:
        meta_state_dict = model_args.load_state_dict()
    except Exception as exc:
        pytest.skip(f"Full checkpoint load failed (memory / hub / env): {exc}")

    hf_full = model_args.cached_hf_model
    assert hf_full is not None

    text_root = _text_model_root(hf_full.model)
    rotary = text_root.rotary_emb
    rotary.eval()
    hf_layers = text_root.layers[:n_layers]
    for layer in hf_layers:
        assert isinstance(layer, Ministral3DecoderLayer), type(layer)
        layer.eval()

    rope_params = resolve_rope_parameters(text_cfg)

    tt_ccl = TT_CCL(mesh_device)
    transformation_mats = {"decode": None, "prefill": None}
    weight_cache_path = model_args.weight_cache_path(dtype)

    tt_layers = [
        TtMinistral3DecoderLayer(
            mesh_device,
            tt_ccl,
            model_args,
            meta_state_dict,
            weight_cache_path,
            layer_num=i,
            dtype=dtype,
            transformation_mats=transformation_mats,
            configuration=model_args,
            llama_4_scaling_beta=rope_params.get("llama_4_scaling_beta"),
            original_max_position_embeddings=rope_params.get("original_max_position_embeddings"),
        )
        for i in range(n_layers)
    ]

    tt_rope = TtMinistral3RotaryEmbedding(
        device=mesh_device,
        batch_size=batch_size,
        head_dim=model_args.head_dim,
        max_seq_len=model_args.max_seq_len,
        config=text_cfg,
    )

    hf_cache = DynamicCache()
    generation_start_pos = 0
    all_pass = True

    residual_mem_cfg = model_args.get_residual_mem_config(Mode.DECODE, None)
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    decode_in_dev = ttnn.to_device(
        model_args.prepare_residual_tensor_decode(
            torch.zeros(batch_size, 1, model_args.dim, dtype=torch.bfloat16),
            residual_mem_cfg,
            on_host=True,
        ),
        mesh_device,
        memory_config=residual_mem_cfg,
    )
    pos_host_init = ttnn.from_torch(
        np.zeros((1, batch_size), dtype=np.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=mesh_mapper,
    )
    current_pos_dev = ttnn.to_device(pos_host_init, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(pos_host_init)

    pcc_steps = [] if PROFILE_DECODE else list(range(DECODE_STEPS))
    if PROFILE_DECODE and DECODE_STEPS > 0:
        pcc_steps = [DECODE_STEPS - 1]

    for step in range(DECODE_STEPS):
        pos = generation_start_pos + step
        pt_decode_in = torch.randn(batch_size, 1, model_args.dim, dtype=torch.bfloat16)
        position_ids = torch.full((batch_size, 1), pos, dtype=torch.long)

        ref_out = _hf_decoder_stack_forward(hf_layers, rotary, pt_decode_in, position_ids, hf_cache)

        current_pos = np.full((batch_size,), pos, dtype=np.int32)
        pos_host = ttnn.from_torch(
            current_pos.reshape(1, batch_size),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper,
        )
        ttnn.copy_host_to_device_tensor(pos_host, current_pos_dev)
        ttnn.deallocate(pos_host)

        rot_mats = tt_rope.get_rot_mats(torch.from_numpy(current_pos))

        decode_in_host = model_args.prepare_residual_tensor_decode(
            pt_decode_in,
            residual_mem_cfg,
            on_host=True,
        )
        ttnn.copy_host_to_device_tensor(decode_in_host, decode_in_dev)
        ttnn.deallocate(decode_in_host)

        h_tt = decode_in_dev
        for tt_layer in tt_layers:
            h_tt = tt_layer.forward_decode(h_tt, current_pos_dev, rot_mats)

        if step in pcc_steps:
            ttnn.synchronize_device(mesh_device)
            tt_torch = ttnn.to_torch(h_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
            tt_torch = tt_torch[:, :1, :batch_size, : model_args.dim].reshape(batch_size, 1, model_args.dim)

            passing, pcc_value = comp_pcc(ref_out, tt_torch, PCC_TARGET)
            logger.info(comp_allclose(ref_out, tt_torch))
            logger.info(f"Decode step {step} pos={pos} n_layers={n_layers} PCC: {pcc_value}")
            if not passing:
                all_pass = False

    if PROFILE_DECODE:
        logger.info("MINISTRAL3_DECODER_PROFILE=1: PCC on final step only (no per-step to_torch in Tracy hot path).")

    assert all_pass, f"Decode PCC below {PCC_TARGET} for one or more steps"
