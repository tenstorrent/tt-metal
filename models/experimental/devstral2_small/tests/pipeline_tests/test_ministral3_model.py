# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# PCC: HF Ministral3Model vs TtMinistral3Model (Devstral weights; FP8 compat on import).

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
from models.experimental.devstral2_small.devstral_utils.multimodal_demo_helpers import resolve_rope_parameters
from models.experimental.devstral2_small.tt.pipeline.tt_ministral3_model import TtMinistral3Model
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.model_config import ModelArgs

DEVSTRAL_REPO_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"
MINISTRAL_SHORT_PREFILL_L1_WIDTH_MM_ENV = "TT_MINISTRAL3_SHORT_PREFILL_L1_WIDTH_MM"
_MESH_DEVICE_PRESETS = {"P150": (1, 1), "BH-QB": (1, 4)}


def _mesh_device_param():
    """Mesh shape for parametrized tests without probing PCIe at pytest collection."""
    mesh_env = os.environ.get("MESH_DEVICE")
    if mesh_env in _MESH_DEVICE_PRESETS:
        return _MESH_DEVICE_PRESETS[mesh_env]
    try:
        return ttnn._ttnn.multi_device.SystemMeshDescriptor().shape().mesh_size()
    except Exception:
        return int(os.environ.get("TT_MESH_WIDTH", "4"))


def _text_model_root(multimodal_inner):
    lm = multimodal_inner.language_model
    return lm.model if hasattr(lm, "model") else lm


def _as_tile_mesh_tensor(torch_tensor, mesh_device, cache_stem):
    return ttnn.as_tensor(
        torch_tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        cache_file_name=cache_stem,
    )


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


def _run_ministral3_model_prefill_pcc(
    mesh_device,
    seq_len: int,
    batch_size: int,
) -> None:
    dtype = ttnn.bfloat16
    model_args = ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=max(512, seq_len),
        dummy_weights=False,
        use_hf_rope=True,
        cache_hf=True,
    )
    model_args.is_distributed_norm = types.MethodType(lambda self, mode: False, model_args)

    depth = int(model_args.full_model_n_layers)
    n_layers = depth
    model_args.n_layers = n_layers

    logger.info(
        f"Ministral3 model prefill PCC: n_layers={n_layers}/{depth} " f"seq_len={seq_len} batch_size={batch_size}"
    )

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

    torch.manual_seed(0)
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

    hidden = x
    for layer in text_root.layers[:n_layers]:
        hidden = layer(
            hidden_states=hidden,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
    ref_out = text_root.norm(hidden)

    rope_params = resolve_rope_parameters(text_cfg)

    tt_ccl = TT_CCL(mesh_device)
    transformation_mats = {"decode": None, "prefill": None}
    input_cache_path = model_args.weight_cache_path(dtype) / "pipeline_test_inputs"
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
    cos_tt = _as_tile_mesh_tensor(
        cos.unsqueeze(0),
        mesh_device,
        input_cache_path / "cos",
    )
    sin_tt = _as_tile_mesh_tensor(
        sin.unsqueeze(0),
        mesh_device,
        input_cache_path / "sin",
    )
    rot_mats = [cos_tt, sin_tt]

    x_tt = _as_tile_mesh_tensor(
        x.unsqueeze(1),
        mesh_device,
        input_cache_path / f"x_bs{batch_size}_seq{seq_len}_dim{model_args.dim}_seed0",
    )
    pos_tt = _as_tile_mesh_tensor(
        position_ids.to(torch.bfloat16).reshape(1, 1, 1, seq_len),
        mesh_device,
        input_cache_path / f"position_ids_bs{batch_size}_seq{seq_len}",
    )

    tt_out = tt_model.forward_prefill_from_embeddings(x_tt, rot_mats, pos_tt)
    tt_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
    if tt_torch.shape != ref_out.shape:
        tt_torch = tt_torch.reshape(ref_out.shape)

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(ref_out, tt_torch, pcc_required)
    logger.info(comp_allclose(ref_out, tt_torch))
    logger.info(f"n_layers={n_layers}/{depth} PCC: {pcc_message}")
    assert passing, f"PCC below {pcc_required} (n_layers={n_layers}/{depth}): {pcc_message}"


@torch.no_grad()
@pytest.mark.timeout(7200)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("mesh_device", [_mesh_device_param()], indirect=True)
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
    """Full text stack prefill PCC (all layers from HF config via ModelArgs.full_model_n_layers)."""
    monkeypatch.setenv("HF_MODEL", DEVSTRAL_REPO_ID)
    monkeypatch.setenv(MINISTRAL_SHORT_PREFILL_L1_WIDTH_MM_ENV, "1")
    _run_ministral3_model_prefill_pcc(mesh_device, seq_len, batch_size)
