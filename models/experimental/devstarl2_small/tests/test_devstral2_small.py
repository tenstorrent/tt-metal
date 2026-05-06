# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: HF ``Mistral3Model.get_image_features`` vs TT ``TtDevstral2SmallModel`` (vision + projector)."""

from __future__ import annotations

import os
import types

import pytest
import torch
from loguru import logger
from transformers.models.mistral3.modeling_mistral3 import Mistral3Model
from transformers.models.pixtral.modeling_pixtral import position_ids_in_meshgrid

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstarl2_small.tt.tt_devstral2_small_model import TtDevstral2SmallModel
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.model_config import ModelArgs

DEVSTRAL_REPO_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"


def _max_patch_grid_side(cfg):
    iz = cfg.image_size
    if isinstance(iz, (tuple, list)):
        iz = iz[0]
    return iz // cfg.patch_size


@pytest.fixture
def trust_remote_ministral(monkeypatch):
    from models.tt_transformers.tt import model_config as mc

    orig_set = mc.ModelArgs._set_hf_params

    def _set_hf_params_trust(self, checkpoint_dir: str):
        self.trust_remote_code_hf = True
        return orig_set(self, checkpoint_dir)

    monkeypatch.setattr(mc.ModelArgs, "_set_hf_params", _set_hf_params_trust)

    def _get_hf_model_cls_devstral_safe(self):
        from transformers.models.auto.modeling_auto import AutoModelForImageTextToText

        if not self.is_multimodal:
            raise ValueError("expected multimodal config")
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
def test_devstral2_small_projected_image_features_pcc(mesh_device, monkeypatch, trust_remote_ministral):
    monkeypatch.setenv("HF_MODEL", DEVSTRAL_REPO_ID)

    dtype_tt = ttnn.bfloat16
    model_args = ModelArgs(
        mesh_device,
        max_batch_size=1,
        max_seq_len=4096,
        dummy_weights=False,
        use_hf_rope=True,
        cache_hf=True,
    )
    model_args.is_distributed_norm = types.MethodType(lambda self, mode: False, model_args)

    try:
        meta_state_dict = model_args.load_state_dict()
    except Exception as exc:
        pytest.skip(f"Full checkpoint load failed (memory / hub / env): {exc}")

    # After load only: ``ModelArgs.load_state_dict`` drops keys containing ``layers.{i}.`` for i≥n_layers.
    # That substring matches ``vision_tower.transformer.layers.{i}`` too, so we must load with full
    # ``n_layers`` first, then trim the **text** decoder depth for ``TtMinistral3Model``.
    model_args.n_layers = 1

    hf_full = model_args.cached_hf_model
    assert hf_full is not None
    hf_inner = hf_full.model
    assert isinstance(hf_inner, Mistral3Model), type(hf_inner)

    vision_cfg = hf_full.config.vision_config
    patch_sz = int(vision_cfg.patch_size)
    H = W = patch_sz * 32
    pixel_values = torch.randn(1, 3, H, W, dtype=torch.bfloat16)
    image_sizes_tensor = torch.tensor([[H, W]], dtype=torch.long)
    image_sizes_list = [(H, W)]

    # Build HF reference like ``Mistral3Model.get_image_features`` (projector output before per-image split).
    # Some checkpoints / wrappers make ``get_image_features`` return a non-tuple; avoid ``torch.cat`` on it.
    cfg = hf_inner.config
    vfl = cfg.vision_feature_layer
    img_out = hf_inner.vision_tower(pixel_values, image_sizes=image_sizes_tensor, output_hidden_states=True)
    hs = img_out.hidden_states
    if isinstance(vfl, int):
        selected = hs[vfl]
    else:
        selected = torch.cat([hs[i] for i in vfl], dim=-1)
    ref = hf_inner.multi_modal_projector(selected.squeeze(0), image_sizes_tensor)
    while ref.dim() > 2:
        ref = ref.squeeze(0)

    hf_vm = hf_inner.vision_tower
    target_dtype = hf_vm.patch_conv.weight.dtype
    pe_conv = hf_vm.patch_conv(pixel_values.to(dtype=target_dtype))
    plist = [e[..., : s[0] // patch_sz, : s[1] // patch_sz] for e, s in zip(pe_conv, image_sizes_list)]
    position_ids = position_ids_in_meshgrid(plist, max_width=_max_patch_grid_side(vision_cfg))
    pos_tt = ttnn.from_torch(
        position_ids.unsqueeze(0).to(torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_model = TtDevstral2SmallModel(
        mesh_device=mesh_device,
        tt_ccl=TT_CCL(mesh_device),
        model_args=model_args,
        meta_state_dict=meta_state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype_tt),
        dtype=dtype_tt,
        transformation_mats={"decode": None, "prefill": None},
        configuration=model_args,
        vision_config=vision_cfg,
    )

    tt_out = tt_model.get_projected_image_features(pixel_values, image_sizes_list, pos_tt)

    tt_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)).float()
    while tt_torch.dim() > 2:
        tt_torch = tt_torch.squeeze(0)
    tt_torch = tt_torch[: ref.shape[0], : ref.shape[1]]

    pcc_required = 0.94
    passing, msg = comp_pcc(ref.float(), tt_torch, pcc_required)
    logger.info(comp_allclose(ref.float(), tt_torch))
    logger.info(f"PCC devstral2_small image pipeline: {msg}")
    assert passing, f"PCC below {pcc_required}: {msg}"
