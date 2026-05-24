# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# PCC: HF Mistral3PatchMerger vs TTMistral3PatchMerger (Devstral multimodal weights).

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger
from transformers.models.mistral3.modeling_mistral3 import Mistral3PatchMerger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstarl2_small.tt.tt_patchmerger import TTMistral3PatchMerger
from models.tt_transformers.tt.model_config import ModelArgs

DEVSTRAL_REPO_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"


def _patch_merger_state_dict_prefix(meta_state_dict: dict) -> str:
    suffix = "patch_merger.merging_layer.weight"
    for k in meta_state_dict:
        if k.endswith(suffix):
            return k[: -len("merging_layer.weight")]
    raise AssertionError(f"No {suffix!r} key in meta state dict (got {len(meta_state_dict)} keys).")


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
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
def test_mistral3_patch_merger_pcc_devstral_weights(mesh_device, monkeypatch, trust_remote_ministral):
    monkeypatch.setenv("HF_MODEL", DEVSTRAL_REPO_ID)

    batch_size = 1
    dtype_tt = ttnn.bfloat16
    model_args = ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
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
    assert hf_full is not None, "Expected cached HF model after load_state_dict with cache_hf=True."

    hf_pm = hf_full.model.multi_modal_projector.patch_merger
    assert isinstance(
        hf_pm, Mistral3PatchMerger
    ), f"Expected Mistral3PatchMerger; got {type(hf_pm).__module__}.{type(hf_pm).__name__}"

    prefix = _patch_merger_state_dict_prefix(meta_state_dict)

    patch_size = int(hf_pm.patch_size)
    spatial_merge = int(hf_pm.spatial_merge_size)
    hidden_d = int(hf_full.config.vision_config.hidden_size)

    # Patch grid divisible by spatial_merge (2×2 unfold blocks).
    gh, gw = 32, 32
    h_px, w_px = gh * patch_size, gw * patch_size
    n_tokens = gh * gw

    torch.manual_seed(0)
    x_bf16 = torch.randn(n_tokens, hidden_d, dtype=torch.bfloat16)
    image_sizes = torch.tensor([[h_px, w_px]], dtype=torch.long)

    ref = hf_pm(x_bf16, image_sizes)
    assert ref.dtype == torch.bfloat16
    assert ref.shape == (gh // spatial_merge * gw // spatial_merge, hidden_d)

    tt_pm = TTMistral3PatchMerger(
        mesh_device,
        model_args,
        meta_state_dict,
        prefix,
        weight_cache_path=None,
        dtype=dtype_tt,
    )

    x_tt = ttnn.from_torch(
        x_bf16,
        device=mesh_device,
        dtype=dtype_tt,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_out = tt_pm(x_tt, image_sizes.tolist())

    tt_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)).to(torch.float32)
    while tt_torch.dim() > 2:
        tt_torch = tt_torch.squeeze(0)
    tt_torch = tt_torch[: ref.shape[0], : ref.shape[1]]

    ref_f = ref.float()
    pcc_required = 0.99
    passing, msg = comp_pcc(ref_f, tt_torch, pcc_required)
    logger.info(comp_allclose(ref_f, tt_torch))
    logger.info(f"PCC patch merger: {msg}")
    assert passing, f"PCC below {pcc_required}: {msg}"


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
def test_mistral3_patch_merger_pcc_large_grid_devstral_weights(mesh_device, monkeypatch, trust_remote_ministral):
    """1540px-class grid: 110×110 patches → 55×55 merged rows (3025), exercises WS chunking."""
    monkeypatch.setenv("HF_MODEL", DEVSTRAL_REPO_ID)

    batch_size = 1
    dtype_tt = ttnn.bfloat16
    model_args = ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=4096,
        dummy_weights=False,
        use_hf_rope=True,
        cache_hf=True,
    )

    try:
        meta_state_dict = model_args.load_state_dict()
    except Exception as exc:
        pytest.skip(f"Full checkpoint load failed (memory / hub / env): {exc}")

    hf_full = model_args.cached_hf_model
    hf_pm = hf_full.model.multi_modal_projector.patch_merger
    prefix = _patch_merger_state_dict_prefix(meta_state_dict)

    patch_size = int(hf_pm.patch_size)
    spatial_merge = int(hf_pm.spatial_merge_size)
    hidden_d = int(hf_full.config.vision_config.hidden_size)

    gh, gw = 110, 110
    h_px, w_px = gh * patch_size, gw * patch_size
    n_tokens = gh * gw

    torch.manual_seed(1)
    x_bf16 = torch.randn(n_tokens, hidden_d, dtype=torch.bfloat16)
    image_sizes = torch.tensor([[h_px, w_px]], dtype=torch.long)

    ref = hf_pm(x_bf16, image_sizes)
    assert ref.shape == (gh // spatial_merge * gw // spatial_merge, hidden_d)

    tt_pm = TTMistral3PatchMerger(
        mesh_device,
        model_args,
        meta_state_dict,
        prefix,
        weight_cache_path=None,
        dtype=dtype_tt,
    )

    x_tt = ttnn.from_torch(
        x_bf16,
        device=mesh_device,
        dtype=dtype_tt,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_out = tt_pm(x_tt, image_sizes.tolist())
    tt_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)).to(torch.float32)
    while tt_torch.dim() > 2:
        tt_torch = tt_torch.squeeze(0)
    tt_torch = tt_torch[: ref.shape[0], : ref.shape[1]]

    pcc_required = 0.99
    passing, msg = comp_pcc(ref.float(), tt_torch, pcc_required)
    logger.info(f"PCC patch merger large grid: {msg}")
    assert passing, f"PCC below {pcc_required}: {msg}"
