# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: Hugging Face ``Mistral3MultiModalProjector`` vs TT projector on Devstral weights."""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger
from transformers.models.mistral3.modeling_mistral3 import Mistral3MultiModalProjector

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstarl2_small.tt.tt_multimodal_projector import TTMistral3MultiModalProjector
from models.tt_transformers.tt.model_config import ModelArgs

DEVSTRAL_REPO_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"


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
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
def test_mistral3_multimodal_projector_pcc_devstral_weights(mesh_device, monkeypatch, trust_remote_ministral):
    monkeypatch.setenv("HF_MODEL", DEVSTRAL_REPO_ID)

    dtype_tt = ttnn.bfloat16
    model_args = ModelArgs(
        mesh_device,
        max_batch_size=1,
        max_seq_len=512,
        dummy_weights=False,
        use_hf_rope=True,
        cache_hf=True,
    )

    try:
        meta_state_dict = model_args.load_state_dict()
    except Exception as exc:
        pytest.skip(f"Full checkpoint load failed (memory / hub / env): {exc}")
    hf_full = model_args.cached_hf_model
    assert hf_full is not None

    hf_projector = hf_full.model.multi_modal_projector
    assert isinstance(hf_projector, Mistral3MultiModalProjector), type(hf_projector)
    hf_projector.eval()

    patch_size = int(hf_projector.patch_merger.patch_size)
    spatial_merge = int(hf_projector.patch_merger.spatial_merge_size)
    vision_hidden = int(hf_full.config.vision_config.hidden_size)

    gh, gw = 32, 32
    h_px, w_px = gh * patch_size, gw * patch_size
    n_tokens = gh * gw

    torch.manual_seed(0)
    x_bf16 = torch.randn(n_tokens, vision_hidden, dtype=torch.bfloat16)
    image_sizes = torch.tensor([[h_px, w_px]], dtype=torch.long)
    ref = hf_projector(x_bf16, image_sizes)
    assert ref.shape == (gh // spatial_merge * gw // spatial_merge, model_args.dim)

    tt_projector = TTMistral3MultiModalProjector(
        mesh_device=mesh_device,
        args=model_args,
        state_dict=meta_state_dict,
        weight_cache_path=None,
        dtype=dtype_tt,
        eps=float(hf_projector.norm.variance_epsilon),
    )

    x_tt = ttnn.from_torch(
        x_bf16,
        device=mesh_device,
        dtype=dtype_tt,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_out = tt_projector(x_tt, image_sizes.tolist())

    tt_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)).float()
    while tt_torch.dim() > 2:
        tt_torch = tt_torch.squeeze(0)
    tt_torch = tt_torch[: ref.shape[0], : ref.shape[1]]

    pcc_required = 0.97
    passing, msg = comp_pcc(ref.float(), tt_torch, pcc_required)
    logger.info(comp_allclose(ref.float(), tt_torch))
    logger.info(f"PCC multimodal projector: {msg}")
    assert passing, f"PCC below {pcc_required}: {msg}"
