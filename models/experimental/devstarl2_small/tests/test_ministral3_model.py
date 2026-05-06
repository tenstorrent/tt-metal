# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: Hugging Face ``Ministral3Model`` (text path) vs ``TtMinistral3Model`` on Devstral weights."""

from __future__ import annotations

import os
import types

import pytest
import torch
from loguru import logger
from transformers.masking_utils import create_causal_mask
from transformers.models.ministral3.modeling_ministral3 import Ministral3Model

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstarl2_small.tt.tt_ministral3_model import TtMinistral3Model
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.model_config import ModelArgs

DEVSTRAL_REPO_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"


def _text_model_root(multimodal_inner):
    lm = multimodal_inner.language_model
    return lm.model if hasattr(lm, "model") else lm


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


@torch.no_grad()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (128,))
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
def test_ministral3_model_pcc_devstral_weights(
    mesh_device,
    seq_len,
    batch_size,
    monkeypatch,
    trust_remote_ministral,
):
    monkeypatch.setenv("HF_MODEL", DEVSTRAL_REPO_ID)

    dtype = ttnn.bfloat16
    model_args = ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=max(512, seq_len),
        dummy_weights=False,
        use_hf_rope=True,
        cache_hf=True,
    )
    # Keep this test lightweight while validating full model wiring.
    model_args.n_layers = 1
    model_args.is_distributed_norm = types.MethodType(lambda self, mode: False, model_args)

    text_cfg = model_args.hf_config.text_config

    try:
        meta_state_dict = model_args.load_state_dict()
    except Exception as exc:
        pytest.skip(f"Full checkpoint load failed (memory / hub / env): {exc}")

    hf_full = model_args.cached_hf_model
    assert hf_full is not None

    text_root = _text_model_root(hf_full.model)
    assert isinstance(text_root, Ministral3Model), type(text_root)
    rotary = text_root.rotary_emb
    rotary.eval()

    x = torch.randn(batch_size, seq_len, model_args.dim, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    position_embeddings = rotary(x, position_ids=position_ids)
    causal_mask = create_causal_mask(
        config=text_cfg,
        inputs_embeds=x,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )

    # Reference Ministral3Model path restricted to first n_layers.
    hidden = x
    for layer in text_root.layers[: model_args.n_layers]:
        hidden = layer(
            hidden_states=hidden,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
    ref_out = text_root.norm(hidden)

    rope_params = text_cfg.rope_parameters or {}
    if not isinstance(rope_params, dict):
        rope_params = dict(rope_params)

    tt_ccl = TT_CCL(mesh_device)
    transformation_mats = {"decode": None, "prefill": None}
    tt_model = TtMinistral3Model(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        model_args=model_args,
        meta_state_dict=meta_state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
        transformation_mats=transformation_mats,
        configuration=model_args,
        llama_4_scaling_beta=rope_params.get("llama_4_scaling_beta"),
        original_max_position_embeddings=rope_params.get("original_max_position_embeddings"),
    )

    cos, sin = position_embeddings
    cos_tt = ttnn.from_torch(
        cos.unsqueeze(0),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    sin_tt = ttnn.from_torch(
        sin.unsqueeze(0),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    rot_mats = [cos_tt, sin_tt]

    x_tt = ttnn.from_torch(
        x.unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    pos_tt = ttnn.from_torch(
        position_ids.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_out = tt_model.forward_prefill_from_embeddings(x_tt, rot_mats, pos_tt)
    tt_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
    if tt_torch.shape != ref_out.shape:
        tt_torch = tt_torch.reshape(ref_out.shape)

    pcc_required = 0.90
    passing, pcc_message = comp_pcc(ref_out, tt_torch, pcc_required)
    logger.info(comp_allclose(ref_out, tt_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC below {pcc_required}: {pcc_message}"
