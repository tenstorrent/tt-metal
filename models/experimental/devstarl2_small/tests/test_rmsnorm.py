# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
PCC test: Hugging Face ``Ministral3RMSNorm`` vs TT ``RMSNorm`` (``tt_rmsnorm.py``) on Devstral weights.

Loads the full multimodal checkpoint via ``ModelArgs.load_state_dict()`` like ``test_ministralattn.py``,
compares layer 0 ``input_layernorm`` (meta key ``layers.0.attention_norm.weight``) on random activations.

Requirements:
- Enough host RAM / VRAM for ``transformers`` to load Devstral (~24B parameters / FP8→BF16).
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger
from transformers.models.ministral3.modeling_ministral3 import Ministral3RMSNorm

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstarl2_small.tt.tt_rmsnorm import RMSNorm
from models.tt_transformers.tt.model_config import ModelArgs

DEVSTRAL_REPO_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"


def _text_model_root(multimodal_inner):
    """``Mistral3Model.language_model`` → inner causal stack with ``layers``."""
    lm = multimodal_inner.language_model
    return lm.model if hasattr(lm, "model") else lm


@pytest.fixture
def trust_remote_ministral(monkeypatch):
    """Trust remote code for config/load, and avoid ``AutoModelForVision2Seq`` where needed."""

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
def test_ministral3_rmsnorm_pcc_devstral_weights(
    mesh_device,
    seq_len,
    batch_size,
    monkeypatch,
    trust_remote_ministral,
):
    monkeypatch.setenv("HF_MODEL", DEVSTRAL_REPO_ID)

    dtype_tt = ttnn.bfloat16
    model_args = ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=max(512, seq_len),
        dummy_weights=False,
        use_hf_rope=True,
        cache_hf=True,
    )

    text_cfg = model_args.hf_config.text_config

    try:
        meta_state_dict = model_args.load_state_dict()
    except Exception as exc:
        pytest.skip(f"Full checkpoint load failed (memory / hub / env): {exc}")

    hf_full = model_args.cached_hf_model
    assert hf_full is not None, "Expected cached HF model after load_state_dict with cache_hf=True."

    text_root = _text_model_root(hf_full.model)
    hf_norm = text_root.layers[0].input_layernorm
    assert isinstance(
        hf_norm, Ministral3RMSNorm
    ), f"Expected Ministral3RMSNorm; got {type(hf_norm).__module__}.{type(hf_norm).__name__}"

    hidden_size = model_args.dim
    eps = float(text_cfg.rms_norm_eps)

    weight_key_meta = "layers.0.attention_norm.weight"
    assert weight_key_meta in meta_state_dict, f"Missing {weight_key_meta!r} in meta state dict."

    hf_norm.eval()

    torch.manual_seed(0)
    torch_input = torch.randn(batch_size, 1, seq_len, hidden_size, dtype=torch.bfloat16)
    ref_out = hf_norm(torch_input)

    tt_norm = RMSNorm(
        device=mesh_device,
        dim=hidden_size,
        state_dict=meta_state_dict,
        layer_num=0,
        weight_key="attention_norm",
        weight_dtype=dtype_tt,
        is_distributed=False,
        eps=eps,
        simplified_rms=False,
    )

    mode = "prefill" if seq_len > 32 else "decode"

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        dtype=dtype_tt,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_out = tt_norm(tt_input, mode=mode)
    tt_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    tt_torch = tt_torch[:, :, :, :hidden_size]

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(ref_out, tt_torch, pcc_required)
    logger.info(comp_allclose(ref_out, tt_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC below {pcc_required}: {pcc_message}"
